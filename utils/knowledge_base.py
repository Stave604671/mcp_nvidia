import json
import os
import faiss
import numpy as np
import asyncio

from config import DATA_DIR, QA_CACHE_FILE, FAISS_INDEX_FILE
from utils.gemini_utils import get_embedding


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
    使用异步并行处理嵌入向量。
    """
    if not qa_pairs:
        print("没有问答对可用于构建索引。")
        return

    print("开始生成问答对的嵌入向量...")
    # 提取所有问题用于嵌入
    questions = [item['question'] for item in qa_pairs]

    # 异步并行获取所有问题的嵌入向量
    embeddings_tasks = [get_embedding(q) for q in questions]
    embeddings_list = await asyncio.gather(*embeddings_tasks)

    # 过滤掉空的或失败的嵌入，并转换为 NumPy 数组
    valid_embeddings = [emb for emb in embeddings_list if emb is not None and len(emb) > 0]

    if not valid_embeddings:
        print("没有有效的嵌入向量生成，无法构建 FAISS 索引。")
        return

    embedding_dimension = len(valid_embeddings[0])
    embeddings_np = np.array(valid_embeddings).astype('float32')

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
        query_embedding = await get_embedding(query_text)
        query_embedding_np = np.array([query_embedding]).astype('float32')

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