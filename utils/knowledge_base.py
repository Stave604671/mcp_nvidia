import json
import os
import faiss
import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer  # <--- 新增导入

from config import DATA_DIR, QA_CACHE_FILE, FAISS_INDEX_FILE

# from utils.gemini_utils import get_embedding # <--- 移除此行，因为不再从这里获取嵌入

# 初始化 SentenceTransformer 模型
# 这个模型会在第一次使用时自动下载，如果本地没有
print("正在加载 SentenceTransformer 模型 'sentence-transformers/all-MiniLM-L6-v2'...")
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("SentenceTransformer 模型加载完成。")


def ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_qa_from_json() -> list:
    """从 JSON 文件加载问答对"""
    ensure_data_dir()
    if os.path.exists(QA_CACHE_FILE):
        with open(QA_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_qa_to_json(qa_pairs: list):
    """保存问答对到 JSON 文件"""
    ensure_data_dir()
    with open(QA_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)


def load_faiss_index():
    """加载 FAISS 索引"""
    ensure_data_dir()
    if os.path.exists(FAISS_INDEX_FILE):
        return faiss.read_index(FAISS_INDEX_FILE)
    return None


def save_faiss_index(index):
    """保存 FAISS 索引"""
    ensure_data_dir()
    faiss.write_index(index, FAISS_INDEX_FILE)


async def build_and_save_faiss_index(qa_pairs: list):
    """
    从问答对构建 FAISS 索引并保存。
    使用 SentenceTransformer 模型生成嵌入向量。
    """
    if not qa_pairs:
        print("没有问答对可用于构建索引。")
        return

    print("开始生成问答对的嵌入向量...")
    # 提取所有问题用于嵌入
    questions = [item['question'] for item in qa_pairs]

    # 使用 SentenceTransformer 批量生成嵌入向量
    # SentenceTransformer.encode 默认是同步的，但对于批量处理非常高效
    embeddings_np = embed_model.encode(questions)

    # 确保向量是 float32 类型，FAISS 需要这个类型
    embeddings_np = embeddings_np.astype('float32')

    if embeddings_np.size == 0:
        print("没有有效的嵌入向量生成，无法构建 FAISS 索引。")
        return

    embedding_dimension = embeddings_np.shape[1]  # 获取嵌入向量的维度
    # 也可以用 embed_model.get_sentence_embedding_dimension() 来获取维度

    # 创建 FAISS 索引
    index = faiss.IndexFlatL2(embedding_dimension)  # L2 距离，简单欧氏距离
    index.add(embeddings_np)

    save_faiss_index(index)
    print(f"FAISS 索引构建完成，包含 {len(qa_pairs)} 个条目，并已保存到 {FAISS_INDEX_FILE}")


async def search_faiss_index(query_text: str, k: int = 5) -> list:
    """
    搜索 FAISS 索引，返回最相关的问答对。
    Args:
        query_text: 查询文本。
        k: 返回最相似的 k 个结果。
    Returns:
        最相关的问答对列表。
    """
    qa_pairs = load_qa_from_json()
    faiss_index = load_faiss_index()

    if not qa_pairs or faiss_index is None:
        print("知识库为空或FAISS索引不存在。")
        return []

    try:
        # 使用 SentenceTransformer 生成查询文本的嵌入向量
        query_embedding = embed_model.encode([query_text])
        query_embedding_np = query_embedding.astype('float32')

        # 执行搜索
        distances, indices = faiss_index.search(query_embedding_np, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(qa_pairs):  # 确保索引有效
                results.append(qa_pairs[idx])
        return results
    except Exception as e:
        print(f"FAISS 搜索失败: {e}")
        return []


def knowledge_base_exists() -> bool:
    """检查本地是否存在缓存的知识库文件 (JSON 和 FAISS)"""
    return os.path.exists(QA_CACHE_FILE) and os.path.exists(FAISS_INDEX_FILE)