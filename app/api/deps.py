from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.db.session import get_db
from app.core.security import decode_access_token
from app.models.db import User, Role, Permission, PermissionCode
from fastapi.security.http import HTTPBearer, HTTPBase
from fastapi.security.utils import get_authorization_scheme_param


class HTTPBearer(HTTPBase):
    def __init__(self, auto_error: bool = True):
        super().__init__(scheme="bearer", auto_error=auto_error)

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials | None:
        authorization = request.headers.get("Authorization")
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials):
            if self.auto_error:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供认证 Token")
            return None
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)


bearer = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未提供认证 Token")
    return await _verify_user(credentials.credentials, db)


async def _verify_user(token: str, db: AsyncSession) -> User:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效或已过期")
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token 无效")
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == int(user_id))
    )
    user = result.unique().scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已禁用")
    return user


def require_permission(*perm_codes: str, require_all: bool = False):
    """
    权限依赖工厂函数。
    用法:
      @router.get("/...", dependencies=[Depends(require_permission("kb_manage"))])
      @router.get("/...", dependencies=[Depends(require_permission("kb_manage", "chat", require_all=False))])
                                   ^ 默认 OR：有任一权限即放行
    """
    async def dep(
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ):
        # 超级管理员拥有所有权限
        if current_user.is_admin:
            return current_user

        # 加载用户角色和权限
        perm_codes_set = set(perm_codes)
        user_perms = await _get_user_permissions(db, current_user.id)
        user_perm_codes = {p.code for p in user_perms}

        # kb_manage 隐含所有知识库子权限：有 kb_manage 菜单权限，或者有任何 kb:{id} 权限
        kb_codes = {p.code for p in user_perms if p.type == "kb" and p.code.startswith("kb:")}
        if "kb_manage" in user_perm_codes or kb_codes:
            user_perm_codes.update(["kb_manage", "kb_read", "kb_write", "kb_delete", "doc_upload", "doc_delete"])

        # 超级管理员角色也拥有所有权限
        has_admin_role = any(r.is_admin for r in current_user.roles)
        if has_admin_role:
            return current_user

        if require_all:
            missing = perm_codes_set - user_perm_codes
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"无权访问该资源（缺少权限: {', '.join(missing)}）",
                )
        else:
            # 任一权限即可
            #判断两个集合是否 完全不相交 （没有任何交集）
            if perm_codes_set and perm_codes_set.isdisjoint(user_perm_codes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"无权访问该资源（缺少权限: {', '.join(perm_codes_set - user_perm_codes)}）",
                )
        return current_user
    return dep


async def require_kb_access(kb_id: int, db: AsyncSession, user: User) -> bool:
    """
    检查用户是否有权访问指定知识库。
    超级管理员或有对应 KB 权限的用户可访问。
    """
    if user.is_admin:
        return True

    # 检查用户是否有 admin 角色
    if any(r.is_admin for r in user.roles):
        return True

    # 检查该知识库是否在用户的知识库权限列表中
    user_perms = await _get_user_permissions(db, user.id)
    kb_perm_codes = {p.code for p in user_perms if p.code.startswith("kb:")}
    if f"kb:{kb_id}" in kb_perm_codes:
        return True

    return False


async def _get_user_permissions(db: AsyncSession, user_id: int) -> list[Permission]:
    """获取用户所有权限（含角色继承的），优先从 Redis 缓存读取"""
    from app.core.redis_client import cache_get_json, cache_set_json
    from app.core.config import settings as app_settings

    cache_key = f"user_perms:{user_id}"
    cached = await cache_get_json(cache_key)
    if cached:
        perms = []
        for p_data in cached:
            perm = Permission(id=p_data["id"], code=p_data["code"], name=p_data["name"], type=p_data["type"])
            perms.append(perm)
        return perms

    user_result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == user_id)
    )
    user = user_result.unique().scalar_one_or_none()
    if not user:
        return []
    perms = []
    perm_data = []
    for role in user.roles:
        for perm in role.permissions:
            if perm not in perms:
                perms.append(perm)
                perm_data.append({"id": perm.id, "code": perm.code, "name": perm.name, "type": perm.type})

    await cache_set_json(cache_key, perm_data, ttl=app_settings.PERMISSION_CACHE_TTL)
    return perms


async def get_user_kb_ids(db: AsyncSession, user: User) -> list[int]:
    """获取用户有权限访问的知识库 ID 列表（仅限 kb 类型的权限）"""
    user_perms = await _get_user_permissions(db, user.id)
    kb_ids = []
    for p in user_perms:
        if p.type == "kb" and p.code.startswith("kb:"):
            try:
                kb_ids.append(int(p.code.split(":")[1]))
            except (ValueError, IndexError):
                pass
    return kb_ids


async def invalidate_user_permission_cache(user_id: int):
    try:
        from app.core.redis_client import cache_delete
        await cache_delete(f"user_perms:{user_id}")
    except Exception:
        pass


async def invalidate_all_permission_cache():
    try:
        from app.core.redis_client import cache_delete_pattern
        await cache_delete_pattern("user_perms:*")
    except Exception:
        pass


async def get_current_user_optional(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    用于文件类接口：优先从 Header Authorization 获取 token，
    未带则尝试从 query token 参数获取，均无则返回 None。
    """
    token = request.query_params.get("token")
    authorization = request.headers.get("Authorization")
    scheme, credentials = get_authorization_scheme_param(authorization)

    if authorization and scheme and credentials:
        return await _verify_user(credentials, db)
    elif token:
        return await _verify_user(token, db)
    else:
        return None
