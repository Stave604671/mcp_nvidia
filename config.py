import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# --- Gemini API 配置 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_GENERATION_MODEL = "gemini-2.5-flash-preview-05-20"  # 用于生成问答和评估
GEMINI_EMBEDDING_MODEL = "models/embedding-001" # 用于生成嵌入向量

# --- 知识库文件路径 ---
DATA_DIR = "data"
QA_CACHE_FILE = os.path.join(DATA_DIR, "qa_cache.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

# --- FastAPI 后端地址 ---
FASTAPI_BASE_URL = "http://localhost:8000"

# --- 评估模式配置 ---
ASSESSMENT_QUESTION_COUNT = 5 # 考核模式下每次获取的题目数量