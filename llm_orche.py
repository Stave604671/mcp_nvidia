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
FASTAPI_MCP_URL = os.getenv("FASTAPI_MCP_URL", "http://localhost:8000/mcp")

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
            "你是一位多才多艺的AI助手。当用户提出问题或需求时，你首先会尝试利用可用的工具。 "
            "可用的工具包括：\n"
            "- `process_pdf_for_qa`: 用于处理上传的PDF文件，从中提取问答对并构建知识库。当你需要用户提供文件来创建或更新知识库时，可以使用此工具。\n"
            "- `retrieve_knowledge_for_chat`: 用于从知识库中检索与用户查询最相关的问答对。当用户提问且可能需要知识库内容来回答时，调用此工具。\n"
            "- `get_assessment_questions_from_knowledge_point`: 根据用户指定的知识点获取考核题目。当用户希望进行考核时，可以使用此工具来获取问题。\n"
            "- `evaluate_user_response`: 用于评估用户对一个问题的回答质量。在考核模式下，当用户提交答案后，可以使用此工具进行评分和反馈。\n"
            "始终提供清晰、准确、简洁的回答。如果工具返回的信息不足以回答问题，请使用你的通用知识进行补充。当用户要求上传文件或者开始考核时，明确告知用户工具的使用方式，并指导他们提供必要的信息。"
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
    return await query_message("少年人，帮我找一个最合适的工具")


if __name__ == "__main__":
    res = asyncio.run(main_test())