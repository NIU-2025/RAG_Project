import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from app.core.config import settings
from app.core.logger import logger
from typing import List, Dict, Any, Optional
from app.services.langchain.embeddings import LangChainEmbeddingService


class LangChainVectorStore:
    _client: Optional[chromadb.PersistentClient] = None

    @classmethod
    def get_client(cls) -> chromadb.PersistentClient:
        """获取ChromaDB客户端"""
        if cls._client is None:
            cls._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB 已连接，持久化目录: {settings.CHROMA_PERSIST_DIR}")
        return cls._client

    @classmethod
    def get_collection(cls, kb_id: int) -> Chroma:
        """获取知识库对应的向量存储集合"""
        client = cls.get_client()
        collection_name = f"kb_{kb_id}"

        # 获取嵌入模型
        embeddings = LangChainEmbeddingService.get_embeddings()

        # 获取或创建集合
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
            client=client
        )

    @classmethod
    def add_documents(
        cls,
        kb_id: int,
        documents: List[Document],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """添加文档到向量存储"""
        vectorstore = cls.get_collection(kb_id)
        vectorstore.add_documents(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    @classmethod
    def query(
        cls,
        kb_id: int,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """查询向量存储"""
        vectorstore = cls.get_collection(kb_id)
        results = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k,
            filter=where
        )

        # 转换为与原系统兼容的格式
        formatted_results = []
        for doc in results:
            metadata = doc.metadata or {}
            formatted_results.append({
                "doc_id": metadata.get("doc_id", 0),
                "filename": metadata.get("filename", ""),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_num": metadata.get("page_num"),
                "content": doc.page_content,
                "score": 1.0,  # LangChain不直接返回分数，使用1.0作为占位符
                "tags": metadata.get("tags")
            })

        return formatted_results

    @classmethod
    def delete_by_doc_id(cls, kb_id: int, doc_id: int):
        """删除指定文档的所有分块"""
        vectorstore = cls.get_collection(kb_id)
        # LangChain的Chroma实现不支持直接按doc_id删除，需要先查询再删除
        results = vectorstore.similarity_search(
            query="",
            filter={"doc_id": doc_id}
        )
        if results:
            ids_to_delete = [doc.metadata.get("id") for doc in results]
            vectorstore.delete(ids=ids_to_delete)

    @classmethod
    def delete_collection(cls, kb_id: int):
        """删除整个知识库集合"""
        client = cls.get_client()
        collection_name = f"kb_{kb_id}"
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    @classmethod
    def count(cls, kb_id: int) -> int:
        """获取知识库中的文档数量"""
        try:
            vectorstore = cls.get_collection(kb_id)
            return vectorstore._collection.count()
        except Exception:
            return 0

    @classmethod
    def as_retriever(cls, kb_id: int, search_kwargs: Optional[Dict] = None):
        """获取检索器"""
        vectorstore = cls.get_collection(kb_id)
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": 5}
        )