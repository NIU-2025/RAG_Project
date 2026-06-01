from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean,
    ForeignKey, BigInteger, Enum, func
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


class Permission(Base):
    """权限表：定义所有权限"""
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(64), unique=True, nullable=False, index=True)   # 权限编码，如 kb_read
    name = Column(String(128), nullable=False)                          # 权限名称，如 "查看知识库"
    type = Column(String(32), nullable=False, default="menu")           # menu=菜单权限, kb=知识库权限
    created_at = Column(DateTime, server_default=func.now())

    '''
    在数据库中， Permission （权限）和 Role （角色）是典型的 多对多关系 ：
    - 一个角色可以拥有多个权限
    - 一个权限可以被多个角色拥有
    relationship 就是用来在 ORM 层面表达这种关系，让你可以通过对象属性直接访问关联数据。
    secondary="role_permissions" 多对多关系的中间表名称（存储角色和权限的关联关系）
    back_populates="permissions" 指定反向关联的属性名（Role 类中定义的 permissions 属性）
    '''
    roles = relationship("Role", secondary="role_permissions", back_populates="permissions")


class Role(Base):
    """角色表"""
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), unique=True, nullable=False, index=True)
    description = Column(String(255), nullable=True)
    is_admin = Column(Boolean, default=False)     # 超级管理员角色（拥有所有权限）
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    users = relationship("User", secondary="user_roles", back_populates="roles")
    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")


class UserRole(Base):
    """用户角色关联表"""
    __tablename__ = "user_roles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        # 联合唯一约束：同一用户不能重复关联同一角色
        # SQLAlchemy 2.0 联合唯一约束（MySQL 语法）
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )


class RolePermission(Base):
    """角色权限关联表"""
    __tablename__ = "role_permissions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role_id = Column(Integer, ForeignKey("roles.id", ondelete="CASCADE"), nullable=False)
    permission_id = Column(Integer, ForeignKey("permissions.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},
    )


# ===== Permission constants =====
class PermissionCode:
    KB_MANAGE = "kb_manage"        # 知识库管理（包含查看/创建/编辑/删除/上传/删除全部）
    CHAT = "chat"                  # 问答对话
    SEARCH = "search"              # 检索
    STATS = "stats"               # 统计
    MODEL_CONFIG = "model_config"     # 模型配置
    VOICE_CONFIG = "voice_config"     # 语音配置
    USER_MANAGE = "user_manage"       # 用户管理
    ROLE_MANAGE = "role_manage"       # 角色管理


class MenuItem:
    """菜单项定义，供前端和权限过滤使用"""
    MENU_DASHBOARD = "dashboard"
    MENU_KB = "kb"
    MENU_CHAT = "chat"
    MENU_MODELS = "models"
    MENU_VOICE = "voice"
    MENU_USERS = "users"
    MENU_ROLES = "roles"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    roles = relationship("Role", secondary="user_roles", back_populates="users")
    knowledge_bases = relationship("KnowledgeBase", back_populates="owner", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

    @property
    def is_admin(self) -> bool:
        """判断用户是否为超级管理员（拥有 is_admin=True 的角色）"""
        if not self.roles:
            return False
        return any(r.is_admin for r in self.roles)


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False)
    doc_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    owner = relationship("User", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")


class DocumentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)
    file_size = Column(BigInteger, default=0)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.pending)
    error_msg = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0)
    tags = Column(String(512), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(Integer, nullable=False, index=True)
    chroma_id = Column(String(128), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, default=0)
    page_num = Column(Integer, nullable=True)
    token_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())

    document = relationship("Document", back_populates="chunks")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True)
    title = Column(String(255), nullable=True)
    llm_provider = Column(String(32), default="openai")
    llm_model = Column(String(64), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    session = relationship("ChatSession", back_populates="messages")


class ModelConfig(Base):
    """模型配置表：存储各 Provider 的 API Key 等信息，优先级高于 .env 配置"""
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(32), unique=True, nullable=False)  # openai/deepseek/dashscope/qianfan/ollama
    api_key = Column(String(512), nullable=True)       # 主密钥（qianfan 用作 access_key）
    api_secret = Column(String(512), nullable=True)    # 仅 qianfan secret_key
    base_url = Column(String(256), nullable=True)      # 自定义 API 地址
    model_name = Column(String(128), nullable=True)    # 默认模型名
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class VoiceProvider(str, enum.Enum):
    baidu = "baidu"
    aliyun = "aliyun"


class VoiceConfig(Base):
    """语音识别配置表：存储各 Provider 的 API 凭证"""
    __tablename__ = "voice_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(32), unique=True, nullable=False)  # baidu / aliyun
    api_key = Column(String(512), nullable=True)       # 百度 AppID / 阿里 AccessKey ID
    api_secret = Column(String(512), nullable=True)    # 百度 API Key / 阿里 AccessKey Secret
    extra_params = Column(Text, nullable=True)         # JSON，存储 provider 专属参数
    is_enabled = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)        # 是否为默认 Provider
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
