from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional
import os


class Settings(BaseSettings):
    # 应用
    APP_ENV: str = "development"
    SECRET_KEY: str = "dev-secret-key-please-change"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7天

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # 速率限制
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_AUTH: str = "10/minute"
    RATE_LIMIT_UPLOAD: str = "20/minute"

    # Redis
    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    REDIS_ENABLED: bool = True
    BM25_CACHE_TTL: int = 3600
    PROVIDER_CACHE_TTL: int = 300
    PERMISSION_CACHE_TTL: int = 600

    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "rag_system"

    #@property 是 Python 的内置装饰器，用于将类的 方法转换为属性 访问，而不需要调用方法
    # 例如：settings.DATABASE_URL 而不是 settings.DATABASE_URL()
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+aiomysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # 文件上传
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 100
    # 图片文件上传
    MAX_IMAGE_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: str = "png,jpg,jpeg,gif,bmp,webp"
    # 语音文件上传
    MAX_AUDIO_SIZE_MB: int = 30
    ALLOWED_AUDIO_TYPES: str = "mp3,wav,m4a,aac,ogg,wma,flac,pcm"

    # Embedding
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    # 推理设备（仅支持 CPU）
    EMBEDDING_DEVICE: str = "cpu"

    # 大模型
    DEFAULT_LLM_PROVIDER: str = "openai"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://xiaoai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    DASHSCOPE_API_KEY: Optional[str] = None
    DASHSCOPE_MODEL: str = "qwen-max"

    QIANFAN_ACCESS_KEY: Optional[str] = None
    QIANFAN_SECRET_KEY: Optional[str] = None
    QIANFAN_MODEL: str = "ERNIE-4.0-8K"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    # LM Studio
    LMSTUDIO_BASE_URL: str = "http://localhost:1234/v1"
    LMSTUDIO_MODEL: str = ""

    # 检索
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_MULTIPLIER: int = 4
    RERANK_TOP_K: int = 5

    # 分块
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

# 确保必要目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
