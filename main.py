from fastapi import FastAPI, Depends, HTTPException, Path, Query, File, UploadFile
from fastapi.responses import JSONResponse
from fastmcp import FastMCP
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import os

# 从 utils.gemini_utils 导入必要的函数，移除 get_embedding
from utils.gemini_utils import extract_qa_from_pdf, evaluate_answer, chat_with_gemini
# 从 utils.knowledge_base 导入知识库管理函数
from utils.knowledge_base import (
    load_qa_from_json, save_qa_to_json, build_and_save_faiss_index,
    search_faiss_index, knowledge_base_exists, ensure_data_dir
)
from config import ASSESSMENT_QUESTION_COUNT

# 创建数据目录
ensure_data_dir()

# 创建FastAPI应用
app = FastAPI(
    title="知识库问答与考核API",
    description="使用FastAPI和FastMCP构建的基于PDF的知识库系统",
    version="1.0.0"
)




# --- Pydantic 模型 ---
class QAPair(BaseModel):
    question: str
    answer: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class AssessmentStartRequest(BaseModel):
    knowledge_point: str


class AssessmentEvaluateRequest(BaseModel):
    original_question: str
    original_answer: str
    user_answer: str


class EvaluationResult(BaseModel):
    score: int
    feedback: str


class AssessmentQuestion(BaseModel):
    question: str
    original_answer: str  # 包含原始答案用于评估


# --- FastAPI 路由 (API Endpoints) ---

@app.post("/upload_pdf", tags=["知识库管理"])
async def upload_pdf_and_process(file: UploadFile = File(...)):
    """
    上传PDF文件，从中提取问答对，并构建本地知识库（JSON和FAISS索引）。
    如果已存在知识库，将覆盖。
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="只支持PDF文件上传。")

    try:
        pdf_bytes = await file.read()

        # 1. 使用 Gemini 从 PDF 提取 Q&A
        qa_pairs = await extract_qa_from_pdf(pdf_bytes)

        if not qa_pairs:
            raise HTTPException(status_code=500, detail="未能从PDF中提取到问答对，请尝试更换PDF或调整文档内容。")

        # 2. 缓存问答对到本地 JSON
        save_qa_to_json(qa_pairs)

        # 3. 将问答对转换为向量并构建 FAISS 索引
        await build_and_save_faiss_index(qa_pairs)

        return {"message": f"成功从 '{file.filename}' 提取 {len(qa_pairs)} 个问答对，并已构建知识库。"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理PDF文件失败: {str(e)}")


@app.get("/knowledge_base_status", tags=["知识库管理"])
async def get_knowledge_base_status():
    """
    检查本地是否存在缓存的知识库 (JSON 和 FAISS 索引)。
    """
    return {"exists": knowledge_base_exists()}


@app.post("/chat", tags=["通用对话"])
async def chat_with_llm(request: ChatRequest) -> ChatMessage:
    """
    根据知识库状态进行通用对话或知识库问答。
    如果知识库存在，将尝试从知识库中检索相关信息作为上下文。
    """
    user_message = request.messages[-1].content  # 获取最新一条用户消息

    if knowledge_base_exists():
        # 尝试从知识库检索
        retrieved_qa_pairs = await search_faiss_index(user_message, k=3)  # 检索3个最相关的Q&A

        if retrieved_qa_pairs:
            context = "\n\n作为参考的知识点:\n" + "\n".join(
                [f"Q: {qa.get('question')}\nA: {qa.get('answer')}" for qa in retrieved_qa_pairs])
            print(f"Chat with context: {context}")
            # 构建带上下文的对话
            # 这里将 system 消息和 user 消息合并为一个用户消息，或者单独处理
            # 考虑到 chat_with_gemini 现在只接受 List[str]，我们需要将这些字典转换为字符串列表
            messages_for_gemini = [
                "你是一位知识渊博的助手。当用户提问时，如果相关知识点在提供的参考知识中，请优先利用这些知识来回答。如果参考知识点不包含答案，请使用你的通用知识。",
                f"{context}\n\n我的问题是: {user_message}"
            ]
        else:
            print("Chat without specific knowledge base context.")
            # 没有相关知识点，直接通用对话
            messages_for_gemini = [
                "你是一位友好的通用助手。",
                user_message
            ]
    else:
        print("Chat without knowledge base.")
        # 知识库不存在，直接通用对话
        messages_for_gemini = [
            "你是一位友好的通用助手。",
            user_message
        ]

    try:
        # 传入 List[str] 到 chat_with_gemini
        response_text = await chat_with_gemini(messages_for_gemini)
        return ChatMessage(role="model", content=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"与LLM交互失败: {str(e)}")


@app.post("/assessment/start", response_model=List[AssessmentQuestion], tags=["考核模式"])
async def start_assessment(request: AssessmentStartRequest):
    """
    根据用户输入的知识点，从知识库中获取最相关的问答对作为考核题目。
    """
    if not knowledge_base_exists():
        raise HTTPException(status_code=404, detail="知识库不存在，无法开始考核。请先上传PDF构建知识库。")

    qa_pairs = load_qa_from_json()
    if not qa_pairs:
        raise HTTPException(status_code=500, detail="知识库内容为空，无法开始考核。")

    # 根据知识点搜索最相关的问答对
    relevant_qa = await search_faiss_index(request.knowledge_point, k=ASSESSMENT_QUESTION_COUNT * 2)  # 多检索一些，防止重复或质量不高

    if not relevant_qa:
        raise HTTPException(status_code=404, detail="未找到与该知识点相关的问答对。请尝试其他知识点。")

    # 简单去重并限制数量
    assessment_questions = []
    seen_questions = set()
    for qa in relevant_qa:
        if qa['question'] not in seen_questions:
            assessment_questions.append(AssessmentQuestion(question=qa['question'], original_answer=qa['answer']))
            seen_questions.add(qa['question'])
            if len(assessment_questions) >= ASSESSMENT_QUESTION_COUNT:
                break

    if not assessment_questions:
        raise HTTPException(status_code=404, detail="尽管进行了搜索，但未能找到足够的独特考核题目。")

    return assessment_questions


@app.post("/assessment/evaluate", response_model=EvaluationResult, tags=["考核模式"])
async def evaluate_user_answer(request: AssessmentEvaluateRequest):
    """
    评估用户对特定问题的回答质量。
    """
    try:
        eval_result = await evaluate_answer(
            request.original_question,
            request.original_answer,
            request.user_answer
        )
        return EvaluationResult(score=eval_result['score'], feedback=eval_result['feedback'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"评估用户回答失败: {str(e)}")


# --- FastMCP 工具---
# 创建FastMCP服务器
mcp = FastMCP.from_fastapi(app)

# @mcp.tool()
# async def process_pdf_for_qa(query, pdf_data: bytes) -> Dict:
#     """
#     [FastMCP Tool] 处理上传的PDF，提取问答对并构建知识库。
#     此工具供FastMCP代理调用，功能与 /upload_pdf 路由类似。
#     """
#     try:
#         qa_pairs = await extract_qa_from_pdf(pdf_data)
#         save_qa_to_json(qa_pairs)
#         await build_and_save_faiss_index(qa_pairs)
#         return {"status": "success", "message": f"成功提取 {len(qa_pairs)} 个问答对并构建知识库。"}
#     except Exception as e:
#         return {"status": "error", "message": f"处理PDF失败: {str(e)}"}


@mcp.tool()
async def retrieve_knowledge_for_chat(query: str) -> List[QAPair]:
    """
    [FastMCP Tool] 从知识库中检索与查询最相关的问答对。
    此工具可供FastMCP代理在回答用户问题时作为检索增强生成（RAG）的一部分调用。
    """
    return await search_faiss_index(query, k=5)


@mcp.tool()
async def get_assessment_questions_from_knowledge_point(knowledge_point: str) -> List[QAPair]:
    """
    [FastMCP Tool] 根据知识点获取考核题目。
    此工具可供FastMCP代理在启动考核模式时调用。
    """
    relevant_qa = await search_faiss_index(knowledge_point, k=ASSESSMENT_QUESTION_COUNT)
    # FastMCP工具返回的数据结构可能需要更通用，这里不包含 original_answer 字段，如果代理需要，需另外设计
    return [QAPair(question=qa['question'], answer=qa['answer']) for qa in relevant_qa]


@mcp.tool()
async def evaluate_user_response(original_q: str, original_a: str, user_a: str) -> EvaluationResult:
    """
    [FastMCP Tool] 评估用户对一个问题的回答。
    此工具可供FastMCP代理在考核模式下评估用户回答时调用。
    """
    eval_result = await evaluate_answer(original_q, original_a, user_a)
    return EvaluationResult(score=eval_result['score'], feedback=eval_result['feedback'])


# 挂载MCP到FastAPI
app.mount("/mcp", mcp.http_app())

# 运行服务器
if __name__ == "__main__":
    mcp.run(transport="stdio")
    # uvicorn.run(app, host="0.0.0.0", port=8005)