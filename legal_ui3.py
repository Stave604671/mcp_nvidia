import streamlit as st
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
from openai import OpenAI
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import json
import logging
import uuid  # 导入 uuid 模块用于生成唯一的 session_id

# ------------------- 配置 -------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 Gemini API（用于 PDF 提取）
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 初始化 DeepSeek API（用于问答）
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 数据库连接池配置
DB_CONFIG = {
    'host': os.getenv("DB_HOST", "localhost"),
    'user': os.getenv("DB_USER", "root"),
    'password': os.getenv("DB_PASSWORD", "123456"),
    'database': os.getenv("DB_NAME", "Legal")
}

DB_POOL = MySQLConnectionPool(pool_name="legal_pool", pool_size=5, **DB_CONFIG)

# SentenceTransformer 模型
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 提取法律知识的 Prompt
prompt_extract = (
    "这是一份包含题目、知识点总结和表格的法律复习文档，请严格按照以下要求提取信息：\n\n"
    "1. 对每一道题目，请详细分析其考点，并指出正确选项（如果有），说明选择该项的依据（如相关法律条文、原则等），不要只给出泛泛的结论。以---来区分段落\n\n"
    "2. 对文档中的知识点总结部分，每一个考点都要有一个知识点总结，请以知识点总结几个字开头然后请逐条清晰提取，去除重复、模糊或空泛表达，确保每条知识点明确、可考。\n\n"
    "3. 对于表格部分，请整理表格中的内容，总结其包含的关键信息（如对比关系、适用条件等)，不要原样复制表格结构。\n\n"
    "4.具体输出格式如下：---（换行）案例分析：（换行）内容（换行）知识点总结（换行）内容---"
    "所有输出内容应为清晰可存储入知识库的结构化总结。"
)

# 配置参数
DOC_ID = os.getenv("DOC_ID", "1")
TOP_K = int(os.getenv("TOP_K", 10))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))


# ------------------- FastMCP 网络搜索工具 -------------------
class WebSearchTool:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    async def web_search_tool(self, query: str) -> str:
        """执行网络搜索并返回相关结果"""
        try:
            response = self.tavily_client.search(
                query,
                max_results=5,
                search_depth="advanced",
                include_answer="advanced"
            )
            return json.dumps(response, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return "网络搜索失败，请稍后重试。"

    async def call_web_search(self, query: str) -> str:
        """调用 web_search_tool 工具"""
        try:
            return await self.web_search_tool(query)
        except Exception as e:
            logger.error(f"MCP web search error: {e}")
            return "网络搜索失败，请稍后重试。"


# ------------------- 数据库初始化 -------------------
def init_db():
    conn = DB_POOL.get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content LONGTEXT,
            vector BLOB,
            file_path TEXT,
            doc_id VARCHAR(255),
            create_at DATETIME
        ) CHARACTER SET utf8mb4;
    ''')
    # 修改 qa_history 表格创建语句，加入 session_id
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_question TEXT,
            matched_context LONGTEXT,
            model_answer LONGTEXT,
            doc_id VARCHAR(255),
            is_current_session TINYINT,
            session_id VARCHAR(255), -- 新增这一列
            created_at DATETIME
        ) CHARACTER SET utf8mb4;
    ''')
    conn.commit()
    cursor.close()
    conn.close()


# ------------------- 检索上下文 -------------------
async def retrieve_context(question: str, doc_id: str, top_k: int = TOP_K) -> str:
    try:
        question_vec = embed_model.encode(question)
        expected_dim = embed_model.get_sentence_embedding_dimension()

        conn = DB_POOL.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, content, vector FROM knowledge_base WHERE doc_id = %s", (doc_id,))
        rows = cursor.fetchall()

        sims = []
        for row_id, content, vec_bytes in rows:
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            if vec.shape[0] != expected_dim:
                logger.error(f"Invalid vector dimension: expected {expected_dim}, got {vec.shape[0]}")
                continue
            sim = np.dot(question_vec, vec) / (np.linalg.norm(question_vec) * np.linalg.norm(vec))
            sims.append((sim, content))

        sims.sort(reverse=True)
        top_contexts = [(sim, content) for sim, content in sims[:top_k] if sim >= SIMILARITY_THRESHOLD]
        matched_context = "\n".join([content for _, content in top_contexts])
        cursor.close()
        conn.close()

        if not top_contexts or any(sim < SIMILARITY_THRESHOLD for sim, _ in sims[:top_k]):
            logger.info("No sufficient context found in database, performing web search...")
            web_tool = WebSearchTool()
            return await web_tool.call_web_search(question)

        return matched_context

    except mysql.connector.Error as e:
        logger.error(f"Database error: {e}")
        web_tool = WebSearchTool()
        return await web_tool.call_web_search(question)
    except Exception as e:
        logger.error(f"Error in retrieve_context: {e}")
        web_tool = WebSearchTool()
        return await web_tool.call_web_search(question)


# ------------------- 历史记录功能 -------------------
def get_qa_history(doc_id: str):
    conn = DB_POOL.get_connection()
    cursor = conn.cursor(dictionary=True)
    # 按照 session_id 和 created_at 排序
    cursor.execute(
        "SELECT id, user_question, model_answer, is_current_session, session_id, created_at FROM qa_history WHERE doc_id = %s ORDER BY session_id DESC, created_at ASC",
        (doc_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    history_sessions_dict = {}
    for row in rows:
        # 如果 session_id 为空，或者 is_current_session 为 1，则视为一个新的会话起点
        # 这里为了确保历史数据兼容性，如果 session_id 为空，并且 is_current_session=1，我们为其生成一个 session_id
        # 对于新的记录，我们在插入时就会确保 session_id 被填充
        current_session_id = row['session_id']
        if not current_session_id and row['is_current_session'] == 1:
            # 兼容处理旧数据，如果旧数据没有 session_id 但标记为新会话，为其生成一个
            current_session_id = str(uuid.uuid4())
            # 可以在这里更新数据库，但为了避免在每次加载时都更新，最好在首次插入时就确保 session_id 存在
            # 暂不在这里更新数据库，只在内存中处理

        if current_session_id not in history_sessions_dict:
            history_sessions_dict[current_session_id] = []
        history_sessions_dict[current_session_id].append(row)

    # 将字典转换为列表，并按会话开始时间（或 session_id 降序）排序，以便最新的会话在前
    history_sessions = list(history_sessions_dict.values())
    # 根据每个会话中第一个问题的 created_at 进行排序（降序），确保最新的会话显示在前面
    history_sessions.sort(key=lambda session: session[0]['created_at'], reverse=True)

    return history_sessions


def delete_qa_session(session_id: str):
    """根据 session_id 删除 qa_history 中的所有相关记录"""
    conn = DB_POOL.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM qa_history WHERE session_id = %s", (session_id,))
        conn.commit()
        st.success(f"✅ 会话 {session_id} 及其记录已成功删除！")
    except mysql.connector.Error as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        st.error(f"❌ 删除会话 {session_id} 失败：{e}")
    finally:
        cursor.close()
        conn.close()


# ------------------- Streamlit 界面 -------------------
st.set_page_config(layout="wide")

st.title("📘 法律知识提取与问答系统")

init_db()

# 初始化 session_state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "你是一位专业的法律助手。"}]
if 'current_session_id' not in st.session_state:  # 新增：存储当前会话的 session_id
    st.session_state.current_session_id = str(uuid.uuid4())  # 为新会话生成一个唯一的ID
if 'first_call' not in st.session_state:
    st.session_state.first_call = True
if 'display_history_session' not in st.session_state:
    st.session_state.display_history_session = None
if 'question_input_value' not in st.session_state:
    st.session_state.question_input_value = ""

# 侧边栏：历史记录
with st.sidebar:
    st.subheader("📜 历史对话")
    history_sessions = get_qa_history(DOC_ID)

    if history_sessions:
        for i, session in enumerate(history_sessions):
            session_start_time = session[0]['created_at'].strftime("%Y-%m-%d %H:%M")
            session_id_to_delete = session[0]['session_id']  # 获取这个会话的 session_id

            with st.expander(f"会话 {i + 1} ({session_start_time})"):
                st.write(f"**开始于:** {session[0]['user_question']}")
                col_view, col_delete = st.columns([1, 1])
                with col_view:
                    if st.button(f"查看会话详情", key=f"view_session_{session_id_to_delete}"):
                        st.session_state.display_history_session = session
                        st.session_state.question_input_value = ""  # 清空输入框内容
                        st.rerun()
                with col_delete:
                    if st.button(f"删除会话", key=f"delete_session_{session_id_to_delete}"):
                        delete_qa_session(session_id_to_delete)
                        # 如果删除的是当前正在查看的历史会话，则返回主界面
                        if st.session_state.display_history_session and st.session_state.display_history_session[0][
                            'session_id'] == session_id_to_delete:
                            st.session_state.display_history_session = None
                        st.rerun()  # 删除后重新加载页面以更新历史记录列表
    else:
        st.info("暂无历史对话记录。")

# 主区域：文件上传和当前对话内容
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 上传法律文档")
    uploaded_file = st.file_uploader("请上传PDF文件", type=["pdf"])

    if uploaded_file:
        with st.spinner("正在分析并提取内容..."):
            file_bytes = uploaded_file.read()
            doc_id = uploaded_file.name.split('.')[0]

            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash-latest",
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                    prompt_extract
                ]
            )

            summary_text = response.text.strip()
            raw_lines = summary_text.splitlines()
            processed_lines = [line.replace('*', '').strip() for line in raw_lines[1:] if line.strip()]
            processed2 = [line.replace('\n', '').replace(" ", "").replace("案例分析：", "") for line in processed_lines
                          if line != '---']

            conn = DB_POOL.get_connection()
            cursor = conn.cursor()
            for seg in processed2:
                vector = embed_model.encode(seg).tobytes()
                cursor.execute('''
                    INSERT INTO knowledge_base (content, vector, file_path, doc_id, create_at)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (seg, vector, uploaded_file.name, doc_id, datetime.now()))

            conn.commit()
            cursor.close()
            conn.close()

            st.success("✅ 成功写入知识库！")
            st.text_area("提取结果预览", value="\n".join(processed2), height=200)

with col2:
    # 新建对话按钮
    if st.button("➕ 新建对话"):
        st.session_state.chat_history = [{"role": "system", "content": "你是一位专业的法律助手。"}]
        st.session_state.first_call = True
        st.session_state.display_history_session = None
        st.session_state.question_input_value = ""
        st.session_state.current_session_id = str(uuid.uuid4())  # 生成新的 session_id
        st.rerun()

    st.subheader("💬 当前对话")

    if st.session_state.display_history_session:
        st.write("---")
        st.write(
            f"**正在查看历史会话 (开始于 {st.session_state.display_history_session[0]['created_at'].strftime('%Y-%m-%d %H:%M')})**")
        for qa in st.session_state.display_history_session:
            st.write(f"**你**：{qa['user_question']}")
            st.write(f"**模型**：{qa['model_answer']}")
            st.markdown("---")
        # 这里保留一个“返回当前对话”按钮，因为删除按钮现在在侧边栏了
        if st.button("返回当前对话", key="return_current_chat_btn"):
            st.session_state.display_history_session = None
            st.session_state.question_input_value = ""
            st.rerun()
    else:
        for msg in st.session_state.chat_history[1:]:
            if msg["role"] == "user":
                display_question = msg['content'].split('问题：')[-1].replace('请回答：', '').strip()
                st.write(f"**你**：{display_question}")
            else:
                st.write(f"**模型**：{msg['content']}")

        st.markdown("---")  # 分隔对话内容和提问框

        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input(
                "请输入您的问题：",
                key="question_input_key",
                value=st.session_state.question_input_value
            )
            submitted = st.form_submit_button("发送")

            if submitted and user_question:
                st.session_state.question_input_value = ""

                with st.spinner("正在检索并生成回答..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    matched_context = loop.run_until_complete(retrieve_context(user_question, DOC_ID))
                    loop.close()

                    prompt = (
                        "请结合以下法律材料或网络搜索结果，基于内容准确回答用户的问题，不要捏造信息，保持用词严谨。\n\n"
                        f"{matched_context}\n\n"
                        f"问题：{user_question}\n"
                        f"请回答："
                    )

                    st.session_state.chat_history.append({"role": "user", "content": prompt})

                    try:
                        response = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=st.session_state.chat_history,
                            stream=False,
                            temperature=0.7
                        )
                        answer = response.choices[0].message.content

                        conn = DB_POOL.get_connection()
                        cursor = conn.cursor()

                        # 确定当前会话的 session_id
                        session_id_to_insert = st.session_state.current_session_id
                        is_first_call_in_session = 1 if st.session_state.first_call else 0

                        cursor.execute('''
                            INSERT INTO qa_history (user_question, matched_context, model_answer, doc_id, is_current_session, session_id, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ''', (
                        user_question, matched_context, answer, DOC_ID, is_first_call_in_session, session_id_to_insert,
                        datetime.now()))
                        conn.commit()
                        cursor.close()
                        conn.close()

                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.session_state.first_call = False

                        if len(st.session_state.chat_history) > 11:
                            st.session_state.chat_history = [st.session_state.chat_history[
                                                                 0]] + st.session_state.chat_history[-10:]

                        st.success("✅ 回答已生成并存入历史记录！")
                        st.rerun()

                    except Exception as e:
                        logger.error(f"API error: {e}")
                        st.error("❌ API 调用失败，请稍后重试。")