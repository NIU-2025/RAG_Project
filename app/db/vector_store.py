import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.logger import logger
from typing import List, Dict, Any, Optional
import time


class VectorStore:
    _client: Optional[chromadb.PersistentClient] = None

    # 初始化 ChromaDB 客户端，确保全局唯一实例，持久化客户端封装
    @classmethod
    def get_client(cls) -> chromadb.PersistentClient:
        if cls._client is None:
            cls._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB 已连接，持久化目录: {settings.CHROMA_PERSIST_DIR}")
        return cls._client

    @classmethod
    def get_collection(cls, kb_id: int) -> chromadb.Collection:
        """每个知识库对应一个 Collection"""
        client = cls.get_client()
        name = f"kb_{kb_id}"
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}, #HNSW索引，使用余弦相似度
        )

    # 添加文档到集合
    @classmethod
    def add_documents(
        cls,
        kb_id: int,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        collection = cls.get_collection(kb_id)
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    @classmethod
    def query(
        cls,
        kb_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict:
        collection = cls.get_collection(kb_id)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        t_query = time.perf_counter()

        '''
        collection.query(**kwargs)是ChromaDB的向量相似度查询，也叫语义搜索
        1. 传入一个 查询向量 （用户问题的嵌入向量）
        2. ChromaDB 在索引中查找与该向量 最相似 的 top_k 个向量
        3. 返回对应的文本内容和相似度分数
        这是一种 基于语义的模糊匹配 ，不是精确的关键词匹配
        '''
        result = collection.query(**kwargs)
        query_ms = (time.perf_counter() - t_query) * 1000
        logger.debug(f"ChromaDB向量检索: {query_ms:.1f}ms | kb_id={kb_id} | n_results={top_k}")
        return result

    # 删除文档
    @classmethod
    def delete_by_doc_id(cls, kb_id: int, doc_id: int):
        collection = cls.get_collection(kb_id)
        collection.delete(where={"doc_id": doc_id})

    # 删除集合
    @classmethod
    def delete_collection(cls, kb_id: int):
        client = cls.get_client()
        try:
            client.delete_collection(f"kb_{kb_id}")
        except Exception:
            pass

    # 统计集合文档数量
    @classmethod
    def count(cls, kb_id: int) -> int:
        try:
            return cls.get_collection(kb_id).count()
        except Exception:
            return 0
