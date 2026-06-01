from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Any
from datetime import datetime
from app.models.db import DocumentStatus

'''
    定义了 API 中的请求和响应模型，用于与前端进行交互。
'''

# ===== Auth =====
class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="用户名, 3-64个字符, 不能包含特殊字符")
    email: EmailStr = Field(..., description="邮箱, 格式为 example@example.com")
    password: str = Field(..., min_length=6, description="密码, 6个字符以上")


class UserLogin(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class UserOut(BaseModel):
    id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    is_active: bool = Field(..., description="是否激活")
    created_at: datetime = Field(..., description="创建时间")

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str = Field(..., description="访问令牌")
    token_type: str = "bearer"
    user: UserOut


# ===== Permission & Role =====
class PermissionOut(BaseModel):
    id: int = Field(..., description="权限ID")
    code: str = Field(..., description="权限编码")
    name: str = Field(..., description="权限名称")
    type: str = Field(..., description="权限类型")

    model_config = {"from_attributes": True} 


class RoleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="角色名称, 1-64个字符")
    description: Optional[str] = None


class RoleUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=64, description="角色名称, 1-64个字符")
    description: Optional[str] = None


class RoleOut(BaseModel):
    id: int = Field(..., description="角色ID")
    name: str = Field(..., description="角色名称")
    description: Optional[str] = Field(None, description="角色描述")
    is_admin: bool = Field(..., description="是否为管理员角色")
    permissions: List[PermissionOut] = Field(..., description="角色关联的权限列表")
    created_at: datetime = Field(..., description="创建时间")

    model_config = {"from_attributes": True}


class RoleAssignReq(BaseModel):
    role_id: int = Field(..., description="角色ID")


class RoleWithKbIds(RoleOut):
    kb_permission_ids: List[int] = []   # 知识库权限关联的 kb_id 列表


# ===== User Management =====
class UserWithRoles(BaseModel):
    id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    is_active: bool = Field(..., description="是否激活")
    created_at: datetime = Field(..., description="创建时间")
    roles: List[RoleOut] = []

    model_config = {"from_attributes": True}


class UserAssignRoleReq(BaseModel):
    role_ids: List[int]


# ===== KnowledgeBase =====
class KBCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, description="知识库名称, 1-128个字符")
    description: Optional[str] = None


class KBUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128, description="知识库名称, 1-128个字符")
    description: Optional[str] = None


class KBOut(BaseModel):
    id: int = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(None, description="知识库描述")
    owner_id: int = Field(..., description="知识库所有者ID")
    is_public: bool = Field(..., description="是否公开")
    doc_count: int = Field(..., description="文档数量")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    model_config = {"from_attributes": True}


# ===== Document =====
class DocumentOut(BaseModel):
    id: int = Field(..., description="文档ID")
    kb_id: int = Field(..., description="知识库ID")
    filename: str = Field(..., description="文档文件名")
    file_type: str = Field(..., description="文档文件类型")
    file_size: int = Field(..., description="文档文件大小")
    status: DocumentStatus = Field(..., description="文档状态")
    error_msg: Optional[str] = Field(None, description="文档状态错误信息")
    chunk_count: int = Field(..., description="文档分块数量")
    tags: Optional[str] = Field(None, description="文档标签")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")

    model_config = {"from_attributes": True}


# ===== Chat =====
class ChatSessionCreate(BaseModel):
    kb_id: Optional[int] = None
    title: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: Optional[str] = None


class ChatSessionOut(BaseModel):
    id: int
    kb_id: Optional[int]
    title: Optional[str]
    llm_provider: str
    llm_model: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    session_id: int = Field(..., description="聊天会话ID")
    message: str = Field(..., description="用户消息")
    stream: bool = Field(..., description="是否开启流式响应")


class ChatMessageOut(BaseModel):
    id: int = Field(..., description="聊天消息ID")
    session_id: int = Field(..., description="聊天会话ID")
    role: str = Field(..., description="角色")
    content: str = Field(..., description="消息内容")
    sources: Optional[str] = Field(None, description="引用文档来源")
    created_at: datetime = Field(..., description="创建时间")

    model_config = {"from_attributes": True}


# ===== Search =====
class SearchRequest(BaseModel):
    kb_id: int = Field(..., description="知识库ID")
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(5, ge=1, le=20, description="返回结果数量")
    score_threshold: float = Field(0.3, ge=0.0, le=1.0, description="分数阈值")
    file_type: Optional[str] = None
    tags: Optional[str] = None


class SearchResult(BaseModel):
    doc_id: int = Field(..., description="文档ID")
    filename: str = Field(..., description="文档文件名")
    file_type: str = Field(..., description="文档文件类型")
    chunk_index: int = Field(..., description="文档分块索引")
    page_num: Optional[int] = Field(None, description="文档分页索引")
    content: str = Field(..., description="文档内容")
    score: float = Field(..., description="文档分数")
    tags: Optional[str] = Field(None, description="文档标签")


class SearchResponse(BaseModel):
    query: str = Field(..., description="搜索查询")
    results: List[SearchResult] = Field(..., description="搜索结果")
    total: int = Field(..., description="总文档数量")


# ===== ModelConfig =====
class ModelConfigUpdate(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    is_enabled: Optional[bool] = None


class ModelConfigOut(BaseModel):
    id: Optional[int] = None
    provider: str
    api_key: Optional[str] = None      # 脱敏后返回
    api_secret: Optional[str] = None   # 脱敏后返回
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    is_enabled: bool = True
    is_configured: bool = False         # 是否已填写必要凭证
    is_available: bool = False          # 是否通过连通性测试

    model_config = {"from_attributes": True}


# ===== VoiceConfig =====
class VoiceConfigUpdate(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    extra_params: Optional[str] = None    # JSON 字符串，存储 provider 专属参数
    is_enabled: Optional[bool] = None
    is_default: Optional[bool] = None


class VoiceConfigOut(BaseModel):
    id: Optional[int] = None
    provider: str
    api_key: Optional[str] = None      # 脱敏后返回
    api_secret: Optional[str] = None   # 脱敏后返回
    extra_params: Optional[str] = None
    is_enabled: bool = True
    is_default: bool = False
    is_configured: bool = False

    model_config = {"from_attributes": True}


# ===== 通用响应 =====
class Resp(BaseModel):
    code: int = 200 
    message: str = "success"
    data: Optional[Any] = None


class PageResp(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[Any] = None
    total: int = 0
    page: int = 1
    page_size: int = 20
