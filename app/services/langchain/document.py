from typing import List, Dict, Any, Optional
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredExcelLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.logger import logger
from app.services.langchain.embeddings import LangChainEmbeddingService
from langchain.schema import Document as LangChainDocument


class LangChainDocumentService:
    """LangChain文档处理服务"""

    @staticmethod
    def load_document(file_path: str, file_type: str) -> List[LangChainDocument]:
        """加载文档并返回LangChain Document对象"""
        loader = None

        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_type == "docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_type in ["xlsx", "xls"]:
            loader = UnstructuredExcelLoader(file_path)
        elif file_type == "csv":
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")

        documents = loader.load()
        logger.info(f"加载文档: {file_path}, 分块数: {len(documents)}")
        return documents

    @staticmethod
    def split_documents(
        documents: List[LangChainDocument],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[LangChainDocument]:
        """分割文档为文本块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        split_docs = text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，生成 {len(split_docs)} 个文本块")
        return split_docs

    @staticmethod
    def prepare_documents_for_storage(
        documents: List[LangChainDocument],
        doc_id: int,
        filename: str,
        file_type: str,
        kb_id: int,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """准备文档元数据用于存储"""
        prepared_docs = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata or {}
            prepared_docs.append({
                "content": doc.page_content,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "chunk_index": i,
                    "page_num": metadata.get("page", None),
                    "kb_id": kb_id,
                    "tags": tags or []
                }
            })

        return prepared_docs

    @staticmethod
    def create_langchain_documents(
        texts: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> List[LangChainDocument]:
        """创建LangChain Document对象"""
        return [
            LangChainDocument(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]

    @staticmethod
    def extract_text_from_document(doc: LangChainDocument) -> str:
        """从文档中提取文本内容"""
        return doc.page_content

    @staticmethod
    def get_document_metadata(doc: LangChainDocument) -> Dict[str, Any]:
        """获取文档元数据"""
        return doc.metadata or {}