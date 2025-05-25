import google.generativeai as genai
from google.generativeai import types
import json
import os
import httpx

from config import GEMINI_API_KEY, GEMINI_GENERATION_MODEL, GEMINI_EMBEDDING_MODEL

# 配置 Gemini API Key
genai.configure(api_key=GEMINI_API_KEY)


def get_gemini_client():
    """获取 Gemini 客户端实例"""
    return genai.GenerativeModel(GEMINI_GENERATION_MODEL)


async def extract_qa_from_pdf(pdf_bytes: bytes) -> list:
    """
    使用 Gemini 从 PDF 中提取问答对。
    Args:
        pdf_bytes: PDF 文件的字节内容。
    Returns:
        一个包含问答对的列表，每个元素是 {'question': '...', 'answer': '...'}.
    Raises:
        ValueError: 如果 Gemini 返回的不是有效的 JSON。
    """
    model = get_gemini_client()

    prompt = """
    请仔细阅读这份文档，并从中提取核心信息，整理成至少10个详细的问答对。
    每个问答对必须包含一个相关的问题和对应的、简洁而准确的答案。
    请将结果以JSON数组的格式返回，每个JSON对象包含'question'和'answer'两个键。
    例如：
    [
        {"question": "什么是FastAPI？", "answer": "FastAPI是一个现代、快速（高性能）的Web框架，用于使用Python 3.7+构建API。"},
        {"question": "FastAPI的优点是什么？", "answer": "主要优点包括：极高的性能、开箱即用的数据验证和序列化、交互式API文档等。"}
    ]
    """

    # 临时保存 PDF 到文件以供 Gemini 读取 (genai.types.Part.from_bytes 需要文件路径)
    temp_pdf_path = "temp_uploaded_doc.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    try:
        response = await model.generate_content_async(
            contents=[
                types.Part.from_bytes(
                    data=open(temp_pdf_path, 'rb').read(),
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )
        # 尝试解析 Gemini 的响应文本为 JSON
        qa_pairs_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        qa_pairs = json.loads(qa_pairs_str)

        # 验证解析后的数据结构
        if not isinstance(qa_pairs, list):
            raise ValueError("Gemini 返回的不是一个 JSON 数组。")
        for item in qa_pairs:
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                raise ValueError("JSON 数组中的元素格式不正确，缺少 'question' 或 'answer' 键。")

        return qa_pairs
    except httpx.ReadTimeout:
        print("Gemini API 调用超时。")
        raise
    except json.JSONDecodeError as e:
        print(f"解析 Gemini 响应失败，响应内容：\n{qa_pairs_str}\n错误：{e}")
        raise ValueError("Gemini 返回的不是有效的 JSON 格式。请尝试重新上传或调整提示。")
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


async def evaluate_answer(original_question: str, original_answer: str, user_answer: str) -> dict:
    """
    使用 Gemini 评估用户对问题的回答。
    Args:
        original_question: 原始问题。
        original_answer: 原始问题的标准答案。
        user_answer: 用户的回答。
    Returns:
        一个字典，包含 'score' (int) 和 'feedback' (str)。
    Raises:
        ValueError: 如果 Gemini 返回的不是有效的 JSON。
    """
    model = get_gemini_client()

    prompt = f"""
    请评估用户对以下问题的回答质量。
    原始问题：{original_question}
    原始答案：{original_answer}
    用户回答：{user_answer}

    请根据用户回答的准确性、完整性和与原始答案的相关性，给出0到100分之间的分数。
    同时，请提供一句简短的评估反馈。
    请以JSON格式返回结果，包含'score'（整数）和'feedback'（字符串）两个键。例如：
    {{"score": 85, "feedback": "回答准确，但可以更详细一些。"}}
    """

    try:
        response = await model.generate_content_async(prompt)
        eval_result_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        eval_result = json.loads(eval_result_str)

        if not isinstance(eval_result, dict) or "score" not in eval_result or "feedback" not in eval_result:
            raise ValueError("Gemini 评估结果格式不正确。")

        return eval_result
    except json.JSONDecodeError as e:
        print(f"解析 Gemini 评估响应失败，响应内容：\n{eval_result_str}\n错误：{e}")
        raise ValueError("Gemini 评估返回的不是有效的 JSON 格式。")


async def get_embedding(text: str) -> list:
    """
    获取文本的嵌入向量。
    Args:
        text: 需要嵌入的文本。
    Returns:
        文本的嵌入向量列表。
    """
    try:
        result = await genai.embed_content_async(model=GEMINI_EMBEDDING_MODEL, content=text)
        return result['embedding']
    except Exception as e:
        print(f"获取嵌入向量失败: {e}")
        raise


async def chat_with_gemini(messages: list) -> str:
    """
    与 Gemini 进行通用对话。
    Args:
        messages: 对话历史，格式为 [{'role': 'user', 'content': '...'}, {'role': 'model', 'content': '...'}]。
    Returns:
        Gemini 的回复文本。
    """
    model = get_gemini_client()
    try:
        response = await model.generate_content_async(messages)
        return response.text
    except Exception as e:
        print(f"与 Gemini 对话失败: {e}")
        raise