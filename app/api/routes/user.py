from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from app.db.session import get_db
from app.models.db import User, Role, UserRole, PermissionCode
from app.models.schemas import UserWithRoles, UserAssignRoleReq, Resp, PageResp
from app.api.deps import get_current_user, require_permission

router = APIRouter(prefix="/users", tags=["用户管理"])


def _build_user_out(user: User) -> UserWithRoles:
    return UserWithRoles(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at,
        roles=[
            # 仅返回角色基础信息（不含权限，避免数据量过大）
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "is_admin": r.is_admin,
                "permissions": [],
                "created_at": r.created_at,
            }
            for r in (user.roles or [])
        ],
    )


# ===== 用户列表 =====
@router.get("", response_model=PageResp)
async def list_users(
    page: int = 1,
    page_size: int = 20,
    keyword: str = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.USER_MANAGE)),
):
    q = select(User)
    if keyword:
        q = q.where(
            (User.username.contains(keyword)) | (User.email.contains(keyword))
        )
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .order_by(User.created_at.desc())
    )
    users = result.unique().scalars().all()

    # roles 已通过 selectinload 加载，无需 refresh
    data = [_build_user_out(user) for user in users]

    return PageResp(data=data, total=total, page=page, page_size=page_size)


# ===== 用户详情 =====
@router.get("/{user_id}", response_model=Resp)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.USER_MANAGE)),
):
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == user_id)
    )
    user = result.unique().scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return Resp(data=_build_user_out(user))


# ===== 删除用户 =====
@router.delete("/{user_id}", response_model=Resp)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.USER_MANAGE)),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能删除自己")

    # 预加载 roles，避免 user.is_admin 属性触发 lazy load（MissingGreenlet）
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles))
        .where(User.id == user_id)
    )
    user = result.unique().scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if user.is_admin:
        raise HTTPException(status_code=403, detail="禁止删除超级管理员账号")

    await db.delete(user)
    # MySQL ondelete="CASCADE" 会自动清理 user_roles 关联表，无需手动处理
    return Resp(message="用户已删除")


# ===== 给用户分配角色 =====
@router.post("/{user_id}/roles", response_model=Resp)
async def assign_roles_to_user(
    user_id: int,
    body: UserAssignRoleReq,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.USER_MANAGE)),
):
    # 禁止给自己分配角色（防止权限混乱）
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能修改自己的角色")

    # 目标用户
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == user_id)
    )
    target_user = result.unique().scalar_one_or_none()
    if not target_user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if target_user.is_admin:
        raise HTTPException(status_code=403, detail="禁止修改超级管理员账号的角色")

    # 验证角色存在
    if body.role_ids:
        roles_result = await db.execute(select(Role).where(Role.id.in_(body.role_ids)))
        roles = list(roles_result.scalars().all())
        if len(roles) != len(body.role_ids):
            raise HTTPException(status_code=400, detail="包含无效的角色 ID")

    # 删除旧的关联
    old_urs = await db.execute(select(UserRole).where(UserRole.user_id == user_id))
    for ur in old_urs.scalars().all():
        await db.delete(ur)

    # 创建新关联
    for role_id in body.role_ids:
        db.add(UserRole(user_id=user_id, role_id=role_id))

    await db.flush()
    # 重新加载用户角色（用 selectinload 代替 db.refresh）
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == user_id)
    )
    target_user = result.unique().scalar_one()

    from app.api.deps import invalidate_user_permission_cache
    await invalidate_user_permission_cache(user_id)

    return Resp(data=_build_user_out(target_user), message="角色分配成功")


# ===== 获取所有角色列表（不含分页，供分配时选择）=====
@router.get("/roles/all", response_model=Resp)
async def list_all_roles_for_assignment(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.USER_MANAGE)),
):
    result = await db.execute(select(Role).order_by(Role.is_admin.desc(), Role.created_at.desc()))
    roles = result.scalars().unique().all()
    data = [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "is_admin": bool(r.is_admin),
            "created_at": r.created_at,
        }
        for r in roles
    ]
    return Resp(data=data)
