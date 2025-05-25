import asyncio
import os
from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp_llm_bridge.config import BridgeConfig, LLMConfig
from mcp_llm_bridge.bridge import BridgeManager, MCPLLMBridge
import logging
from typing import Optional


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()  # 加载环境变量

# --- LLM 和 Bridge 配置 ---
# 确保 FastAPI 应用在 http://0.0.0.0:8000 上运行，并且FastMCP挂载点是 /mcp
# 这个URL是你的FastAPI应用中`app.mount("/mcp", mcp.http_app())`指定的
FASTAPI_MCP_URL = os.getenv("FASTAPI_MCP_URL", "http://localhost:8005/mcp")

# LLM 配置 (从环境变量获取，或者使用默认值)
LLM_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME", "nvidia/llama-3.1-nemotron-ultra-253b-v1")
LLM_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")


async def query_message(user_input: str):
    """
    处理用户输入并返回 LLM 的响应
    """
    if not LLM_API_KEY:
        logger.error("NVIDIA_API_KEY is not set in environment variables.")
        raise ValueError("LLM API Key (NVIDIA_API_KEY) is required.")
    config = BridgeConfig(
        # 使用 HttpServerParameters 连接到已运行的 FastAPI FastMCP 服务器
        mcp_server_params=StdioServerParameters(
            command="python",
            args=["main.py", user_input],
            env=None,
            encoding="utf-8",
        )
        ,
        llm_config=LLMConfig(
            api_key=LLM_API_KEY,
            model=LLM_MODEL_NAME,
            base_url=LLM_BASE_URL
        ),
        system_prompt=(
            "你是一位多才多艺的AI助手，能够理解并执行用户的多种请求。你的首要任务是高效地利用以下可用的工具来提供帮助。当工具无法满足需求时，你会运用你的通用知识进行回答。\n\n"
            "可用的工具 (以及如何使用它们)：\n"
            "\n"
            "- **`retrieve_knowledge_for_chat(query: str)`**\n"
            "  - **用途**: 从已构建的本地知识库中检索与用户问题最相关的问答对。\n"
            "  - **何时使用**: 当用户提出一个需要从知识库中获取具体信息的问题时（例如，关于某个概念、功能、历史事件等）。\n"
            "  - **参数**: `query` (字符串类型)，这是你的搜索关键词，通常是用户的问题本身或其核心内容。\n"
            "  - **输出处理**: 工具将返回一个包含相关问答对的列表。你需要将这些检索到的信息作为上下文，来生成一个准确、全面且直接回答用户问题的响应。如果检索结果为空或不相关，请使用你的通用知识进行回答，并告知用户知识库中可能没有相关信息。\n"
            "\n"
            "- **`get_assessment_questions_from_knowledge_point(knowledge_point: str)`**\n"
            "  - **用途**: 根据用户指定的知识点，从知识库中获取一组考核题目。\n"
            "  - **何时使用**: 当用户明确表示想要“开始考核”或“获取关于某个知识点的题目”时，你应调用此工具。\n"
            "  - **参数**: `knowledge_point` (字符串类型)，用户指定的考核主题或概念。\n"
            "  - **输出处理**: 工具返回一个包含问题和其原始答案的列表。你应该将这些问题清晰地呈现给用户，并指示他们开始回答。此工具通常是考核模式的开始，由用户触发。\n"
            "\n"
            "- **`evaluate_user_response(original_q: str, original_a: str, user_a: str)`**\n"
            "  - **用途**: 评估用户对一个考核问题的回答质量，并提供得分和反馈。\n"
            "  - **何时使用**: **此工具应在你已经知道原始问题、标准答案和用户的回答时才被调用。** 在考核模式下，当用户提交了对某个问题的回答后，为了给出即时评估，你需要精确地提供这三个参数。\n"
            "  - **参数**: \n"
            "    - `original_q` (字符串类型): 考核的原始问题文本。\n"
            "    - `original_a` (字符串类型): 该问题的标准或正确答案。\n"
            "    - `user_a` (字符串类型): 用户提交的回答文本。\n"
            "  - **输出处理**: 工具将返回一个得分和反馈信息。你需要将这些评估结果清晰、友好地告知用户，帮助他们理解自己的掌握程度。\n"
            "\n"
            "**重要说明：**\n"
            "1.  **文件上传**: 如果用户提到要上传PDF文件，请指导他们使用前端界面提供的“上传PDF文件”按钮，因为你没有直接处理文件上传的工具。\n"
            "2.  **通用回答**: 如果用户的请求与上述工具的功能不符，或者工具返回的信息不足以完全回答问题，请运用你的通用知识提供帮助。\n"
            "3.  **清晰沟通**: 始终提供清晰、准确、简洁的回答。在需要时，引导用户进行下一步操作。"
        )
    )
    async with BridgeManager(config) as bridge:
        try:
            response = await bridge.process_message(user_input)
            return response
        except Exception as e:
            logger.error(f"\nError occurred: {e}")


# 用于手动测试 LLM 编排器
async def main_test():
    """手动测试 LLM 编排器功能"""
    print(f"--- 启动 LLM 编排器测试 ---")
    print(f"请确保您的 FastAPI 后端已运行在 {FASTAPI_MCP_URL}!")
    print("----------------------------")
    return await query_message("帮我查找有关刑法案例分析的题目，然后开始考察我的掌握程度。")


if __name__ == "__main__":
    res = asyncio.run(main_test())