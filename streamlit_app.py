import streamlit as st
import httpx
import asyncio

from config import FASTAPI_BASE_URL, ASSESSMENT_QUESTION_COUNT

# FastAPI 后端 URL
BASE_URL = FASTAPI_BASE_URL

st.set_page_config(page_title="知识库问答与考核系统", layout="wide")

st.title("📚 知识库问答与考核系统")
st.markdown("---")


# --- 辅助函数 ---
async def check_knowledge_base_status():
    """检查知识库是否存在"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/knowledge_base_status")
            response.raise_for_status()
            return response.json().get("exists", False)
    except httpx.RequestError as e:
        st.error(f"无法连接到后端服务: {e}")
        return False
    except Exception as e:
        st.error(f"检查知识库状态时发生错误: {e}")
        return False


async def upload_pdf(uploaded_file):
    """上传PDF并触发后端处理"""
    if uploaded_file is None:
        return

    with st.spinner("正在上传并处理PDF，请稍候..."):
        try:
            async with httpx.AsyncClient() as client:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = await client.post(f"{BASE_URL}/upload_pdf", files=files, timeout=600.0)  # 增加超时时间
                response.raise_for_status()
                st.success(response.json().get("message", "PDF处理成功！"))
                st.session_state.knowledge_base_exists = True  # 刷新知识库状态
        except httpx.HTTPStatusError as e:
            st.error(f"PDF处理失败: {e.response.json().get('detail', e.response.text)}")
        except httpx.RequestError as e:
            st.error(f"连接后端失败: {e}")
        except Exception as e:
            st.error(f"发生未知错误: {e}")


async def send_chat_message(user_message):
    """发送聊天消息到后端"""
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    with st.spinner("AI 思考中..."):
        try:
            async with httpx.AsyncClient() as client:
                # 仅发送当前用户消息给后端，后端会处理上下文
                response = await client.post(
                    f"{BASE_URL}/chat",
                    json={"messages": [{"role": "user", "content": user_message}]}
                )
                response.raise_for_status()
                ai_response = response.json()
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response['content']})
        except httpx.HTTPStatusError as e:
            st.error(f"聊天失败: {e.response.json().get('detail', e.response.text)}")
        except httpx.RequestError as e:
            st.error(f"连接后端失败: {e}")
        except Exception as e:
            st.error(f"发生未知错误: {e}")


async def start_assessment_mode(knowledge_point):
    """启动考核模式，获取考核题目"""
    if not knowledge_point.strip():
        st.warning("请输入考核的知识点。")
        return

    with st.spinner("正在准备考核题目..."):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/assessment/start",
                    json={"knowledge_point": knowledge_point}
                )
                response.raise_for_status()
                st.session_state.assessment_questions = response.json()
                st.session_state.current_question_index = 0
                st.session_state.total_assessment_score = 0
                st.session_state.assessment_results = []  # 存储每次评估结果
                st.session_state.assessment_in_progress = True
                st.success(f"已获取 {len(st.session_state.assessment_questions)} 个考核题目。")
        except httpx.HTTPStatusError as e:
            st.error(f"启动考核失败: {e.response.json().get('detail', e.response.text)}")
        except httpx.RequestError as e:
            st.error(f"连接后端失败: {e}")
        except Exception as e:
            st.error(f"发生未知错误: {e}")


async def submit_assessment_answer(user_answer):
    """提交用户答案并评估"""
    current_q_data = st.session_state.assessment_questions[st.session_state.current_question_index]

    with st.spinner("正在评估你的回答..."):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{BASE_URL}/assessment/evaluate",
                    json={
                        "original_question": current_q_data['question'],
                        "original_answer": current_q_data['original_answer'],
                        "user_answer": user_answer
                    }
                )
                response.raise_for_status()
                eval_result = response.json()

                st.session_state.total_assessment_score += eval_result['score']
                st.session_state.assessment_results.append({
                    "question": current_q_data['question'],
                    "user_answer": user_answer,
                    "original_answer": current_q_data['original_answer'],
                    "score": eval_result['score'],
                    "feedback": eval_result['feedback']
                })

                st.session_state.current_question_index += 1

                # 强制刷新UI以显示下一题或结果
                st.rerun()

        except httpx.HTTPStatusError as e:
            st.error(f"评估失败: {e.response.json().get('detail', e.response.text)}")
        except httpx.RequestError as e:
            st.error(f"连接后端失败: {e}")
        except Exception as e:
            st.error(f"发生未知错误: {e}")


# --- Streamlit Session State 初始化 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base_exists" not in st.session_state:
    st.session_state.knowledge_base_exists = False
if "assessment_mode" not in st.session_state:
    st.session_state.assessment_mode = False
if "assessment_in_progress" not in st.session_state:
    st.session_state.assessment_in_progress = False
if "assessment_questions" not in st.session_state:
    st.session_state.assessment_questions = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "total_assessment_score" not in st.session_state:
    st.session_state.total_assessment_score = 0
if "assessment_results" not in st.session_state:
    st.session_state.assessment_results = []

# --- 页面布局 ---

# 侧边栏用于上传PDF和模式切换
with st.sidebar:
    st.header("功能设置")
    uploaded_files = st.file_uploader(
        "上传PDF文件（可多选，将按顺序处理并覆盖现有知识库）",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            asyncio.run(upload_pdf(uploaded_file))
        st.session_state.knowledge_base_exists = asyncio.run(check_knowledge_base_status())  # 再次确认状态

    st.markdown("---")

    # 检查知识库状态
    if not st.session_state.knowledge_base_exists:
        st.warning("当前没有可用的知识库，请上传PDF。")
        # 实时更新知识库状态
        if asyncio.run(check_knowledge_base_status()):
            st.session_state.knowledge_base_exists = True
            st.success("知识库已加载！")
    else:
        st.success("知识库已就绪！")

    st.markdown("---")
    st.header("模式切换")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("进入闲聊模式"):
            st.session_state.assessment_mode = False
            st.session_state.assessment_in_progress = False
            st.rerun()
    with col2:
        if st.button("进入考核模式"):
            if st.session_state.knowledge_base_exists:
                st.session_state.assessment_mode = True
                st.session_state.assessment_in_progress = False  # 重置考核状态
                st.session_state.assessment_questions = []
                st.session_state.current_question_index = 0
                st.session_state.total_assessment_score = 0
                st.session_state.assessment_results = []
                st.rerun()
            else:
                st.warning("请先上传PDF构建知识库才能进入考核模式。")

# --- 主内容区 ---

if st.session_state.assessment_mode:
    st.header("🎯 考核模式")

    if not st.session_state.assessment_in_progress:
        st.write("请在下方输入你希望考核的知识点，然后点击 '开始考核' 按钮。")
        knowledge_point = st.text_input("考核知识点 (例如: 'FastAPI路由', 'MCP框架')")
        if st.button("开始考核", use_container_width=True):
            asyncio.run(start_assessment_mode(knowledge_point))
    else:
        # 考核进行中
        if st.session_state.current_question_index < len(st.session_state.assessment_questions):
            # 显示当前问题
            current_q_data = st.session_state.assessment_questions[st.session_state.current_question_index]
            st.subheader(
                f"问题 {st.session_state.current_question_index + 1}/{len(st.session_state.assessment_questions)}")
            st.markdown(f"**{current_q_data['question']}**")

            user_answer = st.text_area("你的回答:", key=f"answer_{st.session_state.current_question_index}", height=150)
            if st.button("提交回答并进入下一题", use_container_width=True):
                if user_answer.strip():
                    asyncio.run(submit_assessment_answer(user_answer))
                else:
                    st.warning("请输入你的回答。")
        else:
            # 考核结束
            st.subheader("🎉 考核结束！")
            total_questions = len(st.session_state.assessment_questions)
            if total_questions > 0:
                average_score = st.session_state.total_assessment_score / total_questions
                st.metric(label="你的总得分",
                          value=f"{st.session_state.total_assessment_score} / {total_questions * 100} 分")
                st.metric(label="平均得分", value=f"{average_score:.1f} 分")

                mastery_level = ""
                if average_score >= 90:
                    mastery_level = "精通 (Excellent!)"
                elif average_score >= 75:
                    mastery_level = "掌握良好 (Good Grasp)"
                elif average_score >= 60:
                    mastery_level = "基本掌握 (Basic Understanding)"
                else:
                    mastery_level = "需要加强 (Needs Improvement)"

                st.info(f"你对该知识点的掌握程度：**{mastery_level}**")

                st.markdown("### 详细评估结果")
                for i, result in enumerate(st.session_state.assessment_results):
                    st.expander(f"问题 {i + 1}: {result['question']}").markdown(f"""
                        **你的回答:** {result['user_answer']}
                        **标准答案:** {result['original_answer']}
                        **得分:** {result['score']} / 100
                        **评估反馈:** {result['feedback']}
                    """)
            else:
                st.warning("没有可用的考核题目。")

            if st.button("重新开始考核", use_container_width=True):
                st.session_state.assessment_in_progress = False
                st.session_state.assessment_questions = []
                st.session_state.current_question_index = 0
                st.session_state.total_assessment_score = 0
                st.session_state.assessment_results = []
                st.rerun()

else:
    st.header("💬 闲聊模式")

    # 显示聊天历史
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入框
    if prompt := st.chat_input("向AI提问..."):
        asyncio.run(send_chat_message(prompt))