from typing import List, Optional
from app.core.config import settings
from app.core.logger import logger
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.embeddings.base import Embeddings


class LangChainEmbeddingService:
    _embeddings: Optional[Embeddings] = None

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        """获取LangChain嵌入实例"""
        if cls._embeddings is None:
            if settings.EMBEDDING_PROVIDER == "openai":
                cls._embeddings = OpenAIEmbeddings(
                    openai_api_key=settings.OPENAI_API_KEY,
                    openai_api_base=settings.OPENAI_BASE_URL,
                    model=settings.OPENAI_EMBEDDING_MODEL,
                )
            else:
                cls._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            logger.info(f"加载LangChain嵌入模型: {settings.EMBEDDING_MODEL}")
        return cls._embeddings

    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        """嵌入文本列表"""
        if not texts:
            return []
        embeddings = cls.get_embeddings()
        return embeddings.embed_documents(texts)

    @classmethod
    def embed_query(cls, query: str) -> List[float]:
        """嵌入查询文本"""
        embeddings = cls.get_embeddings()
        return embeddings.embed_query(query)