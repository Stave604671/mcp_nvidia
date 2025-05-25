from google import genai
from google.genai import types
import json
import os
import httpx
import pathlib
import asyncio  # <--- 新增导入，用于运行异步测试
from config import GEMINI_API_KEY, GEMINI_GENERATION_MODEL  # <--- 注意这里不再导入 GEMINI_EMBEDDING_MODEL


def get_gemini_client():
    """获取 Gemini 客户端实例"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY 未设置。请在 .env 文件中设置。")
    # 按照您提供的代码，返回 genai.Client 实例
    return genai.Client(api_key=GEMINI_API_KEY)


async def extract_qa_from_pdf(pdf_bytes: bytes):
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

    # 按照您提供的代码，这里是固定的 'file.pdf'。我不再改动它。
    filepath = pathlib.Path(temp_pdf_path)  # <--- 保持您原始的代码逻辑

    try:
        # 按照您提供的代码，使用 model.models.generate_content
        response = model.models.generate_content(
            model=GEMINI_GENERATION_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=filepath.read_bytes(),
                    mime_type='application/pdf',
                ),
                prompt])

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
    except Exception as e:  # 捕获其他可能的异常
        print(f"提取问答对时发生未知错误: {e}")
        raise
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
        # 按照您提供的代码，使用 model.models.generate_content
        response = model.models.generate_content(
            model=GEMINI_GENERATION_MODEL,
            contents=[prompt]
        )
        eval_result_str = response.text.strip().replace("```json\n", "").replace("\n```", "")
        eval_result = json.loads(eval_result_str)

        if not isinstance(eval_result, dict) or "score" not in eval_result or "feedback" not in eval_result:
            raise ValueError("Gemini 评估结果格式不正确。")

        return eval_result
    except json.JSONDecodeError as e:
        print(f"解析 Gemini 评估响应失败，响应内容：\n{eval_result_str}\n错误：{e}")
        raise ValueError("Gemini 评估返回的不是有效的 JSON 格式。")
    except Exception as e:
        print(f"评估用户回答时发生未知错误: {e}")
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
        response = model.models.generate_content(model=GEMINI_GENERATION_MODEL, contents=messages)
        return response.text
    except Exception as e:
        print(f"与 Gemini 对话失败: {e}")
        raise


# --- main 方法和测试用例 ---
async def main():
    print("--- 开始 Gemini 工具函数测试 ---")

    # --- 测试 chat_with_gemini ---
    print("\n>>> 测试 chat_with_gemini (通用对话) <<<")
    chat_messages = [
        "你好，请介绍一下FastAPI。"
    ]
    try:
        chat_response = await chat_with_gemini(chat_messages)
        print(f"Chat Response: {chat_response[:200]}...")  # 打印前200字
    except Exception as e:
        print(f"Chat Test Failed: {e}")

    # --- 测试 evaluate_answer ---
    print("\n>>> 测试 evaluate_answer (评估回答) <<<")
    original_q = "FastAPI的主要优点是什么？"
    original_a = "FastAPI的主要优点包括：极高的性能、开箱即用的数据验证和序列化、交互式API文档（Swagger UI和ReDoc）自动生成、基于标准（OpenAPI, JSON Schema）和类型提示的优势。"

    # 好的回答
    user_a_good = "FastAPI性能很高，有自动文档和数据验证，而且使用了Python类型提示。"
    print(f"\n--- 评估好的回答 ---")
    print(f"原始问题: {original_q}")
    print(f"标准答案: {original_a}")
    print(f"用户回答: {user_a_good}")
    try:
        eval_result_good = await evaluate_answer(original_q, original_a, user_a_good)
        print(f"评估结果 (好): Score={eval_result_good['score']}, Feedback='{eval_result_good['feedback']}'")
    except Exception as e:
        print(f"Evaluation Test (Good) Failed: {e}")

    # 差的回答
    user_a_bad = "FastAPI是用来写网页的，没什么特别的。"
    print(f"\n--- 评估差的回答 ---")
    print(f"原始问题: {original_q}")
    print(f"标准答案: {original_a}")
    print(f"用户回答: {user_a_bad}")
    try:
        eval_result_bad = await evaluate_answer(original_q, original_a, user_a_bad)
        print(f"评估结果 (差): Score={eval_result_bad['score']}, Feedback='{eval_result_bad['feedback']}'")
    except Exception as e:
        print(f"Evaluation Test (Bad) Failed: {e}")

    # --- 测试 extract_qa_from_pdf ---
    # 注意: 要测试此功能，你需要准备一个实际的PDF文件。
    # 我这里会给出一个提示，并尝试读取一个示例PDF（如果存在）。
    # 按照您提供的代码，这里期望有一个名为 'file.pdf' 的文件。
    print("\n>>> 测试 extract_qa_from_pdf (PDF问答提取) <<<")
    expected_pdf_path = r"E:\PycharmProjects\mcp_nvidia\data\刑法案例分析之模拟题汇总.pdf"  # 按照您提供的代码，需要一个名为 'file.pdf' 的文件

    if os.path.exists(expected_pdf_path):
        print(f"正在尝试从 '{expected_pdf_path}' 提取问答对...")
        try:
            with open(expected_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            qa_pairs = await extract_qa_from_pdf(pdf_bytes)
            print(f"成功提取 {len(qa_pairs)} 个问答对。示例第一个问答对:")
            if qa_pairs:
                print(f"Q: {qa_pairs[0]['question']}")
                print(f"A: {qa_pairs[0]['answer']}")
            else:
                print("未提取到任何问答对。")
        except Exception as e:
            print(f"PDF Extraction Test Failed: {e}")
    else:
        print(f"跳过 PDF 问答提取测试：未找到文件 '{expected_pdf_path}'。")
        print(f"要测试此功能，请在当前目录下放置一个有效的PDF文件，并将其命名为 '{expected_pdf_path}'。")

    print("\n--- 所有 Gemini 工具函数测试完成 ---")
    return qa_pairs

if __name__ == "__main__":
    # 使用 asyncio.run() 来运行顶层的异步函数
    res = asyncio.run(main())