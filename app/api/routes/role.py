from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from app.db.session import get_db
from app.models.db import (
    User, Role, Permission, RolePermission, PermissionCode,
    UserRole, KnowledgeBase
)
from app.models.schemas import (
    RoleCreate, RoleUpdate, RoleOut, RoleWithKbIds, PermissionOut, Resp, PageResp
)
from app.api.deps import get_current_user, require_permission, get_user_kb_ids

router = APIRouter(prefix="/roles", tags=["角色管理"])


async def _invalidate_perm_cache():
    try:
        from app.api.deps import invalidate_all_permission_cache
        await invalidate_all_permission_cache()
    except Exception:
        pass

# 从角色权限中提取知识库ID
def _extract_kb_ids(permissions: list) -> list[int]:
    kb_ids = []
    for p in permissions:
        if p.type == "kb" and p.code.startswith("kb:"):
            try:
                kb_ids.append(int(p.code.split(":")[1]))
            except (ValueError, IndexError):
                pass
    return kb_ids


# ===== Helpers =====
# 构建角色输出模型，包含知识库权限ID
def _build_role_out(role: Role, kb_ids: list[int] = None) -> RoleWithKbIds:
    return RoleWithKbIds(
        id=role.id,
        name=role.name,
        description=role.description,
        is_admin=role.is_admin,
        permissions=[PermissionOut.model_validate(p) for p in role.permissions],
        created_at=role.created_at,
        kb_permission_ids=kb_ids or [],
    )


async def _get_role_with_perms(db: AsyncSession, role_id: int) -> Role:
    result = await db.execute(
        select(Role)
        .options(selectinload(Role.permissions))
        .where(Role.id == role_id)
    )
    role = result.unique().scalar_one_or_none()
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    return role

# 获取所有角色列表
# ===== 角色列表 =====
@router.get("", response_model=PageResp)
async def list_roles(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    total = await db.scalar(select(func.count()).select_from(Role))
    result = await db.execute(
        select(Role)
        .options(selectinload(Role.permissions))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .order_by(Role.created_at.desc())
    )
    roles = result.unique().scalars().all()

    # permissions 已通过 selectinload 加载，无需 refresh
    data = [_build_role_out(role, _extract_kb_ids(role.permissions)) for role in roles]

    return PageResp(data=data, total=total, page=page, page_size=page_size)


# ===== 角色详情 =====
@router.get("/{role_id}", response_model=Resp)
async def get_role(
    role_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    role = await _get_role_with_perms(db, role_id)

    # 获取该角色关联的知识库权限（type=kb 的权限，其 code 格式为 "kb:{kb_id}"）
    kb_ids = []
    for p in role.permissions:
        if p.type == "kb" and p.code.startswith("kb:"):
            try:
                kb_ids.append(int(p.code.split(":")[1]))
            except ValueError:
                pass

    return Resp(data=_build_role_out(role, kb_ids))


# ===== 新建角色 =====
@router.post("", response_model=Resp)
async def create_role(
    body: RoleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    # 检查角色名是否重复
    existing = await db.execute(select(Role).where(Role.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="角色名称已存在")

    role = Role(name=body.name, description=body.description, is_admin=False)
    db.add(role)
    await db.flush()
    role = await _get_role_with_perms(db, role.id)
    await _invalidate_perm_cache()
    return Resp(data=_build_role_out(role), message="角色创建成功")


# ===== 更新角色 =====
@router.put("/{role_id}", response_model=Resp)
async def update_role(
    role_id: int,
    body: RoleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    role = await _get_role_with_perms(db, role_id)

    # 禁止修改超级管理员角色
    if role.is_admin:
        raise HTTPException(status_code=403, detail="禁止修改超级管理员角色")

    if body.name is not None:
        # 检查重名
        dup = await db.execute(select(Role).where(Role.name == body.name, Role.id != role_id))
        if dup.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="角色名称已存在")
        role.name = body.name

    if body.description is not None:
        role.description = body.description

    await db.flush()
    role = await _get_role_with_perms(db, role_id)
    await _invalidate_perm_cache()
    return Resp(data=_build_role_out(role), message="角色更新成功")


# ===== 删除角色 =====
@router.delete("/{role_id}", response_model=Resp)
async def delete_role(
    role_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    role = await _get_role_with_perms(db, role_id)

    if role.is_admin:
        raise HTTPException(status_code=403, detail="禁止删除超级管理员角色")

    # 检查是否有用户使用此角色
    user_count = await db.scalar(
        select(func.count()).select_from(UserRole).where(UserRole.role_id == role_id)
    )
    if user_count and user_count > 0:
        raise HTTPException(status_code=400, detail=f"该角色已被 {user_count} 个用户使用，请先解除关联后再删除")

    await db.delete(role)
    await _invalidate_perm_cache()
    return Resp(message="角色已删除")


# ===== 获取所有可用权限列表 =====
@router.get("/permissions/all", response_model=Resp)
async def list_all_permissions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    result = await db.execute(select(Permission).order_by(Permission.type, Permission.id))
    permissions = result.scalars().all()

    # 按类型分组
    menu_perms = [PermissionOut.model_validate(p) for p in permissions if p.type == "menu"]
    # kb 类型的权限动态构建：管理员返回全部知识库，普通用户返回自己拥有 + 角色授权的
    if current_user.is_admin:
        kbs_result = await db.execute(select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()))
    else:
        user_kb_ids = await get_user_kb_ids(db, current_user)
        kbs_result = await db.execute(
            select(KnowledgeBase)
            .where(
                (KnowledgeBase.owner_id == current_user.id)
                | (KnowledgeBase.id.in_(user_kb_ids))
            )
            .order_by(KnowledgeBase.created_at.desc())
        )
    kbs = kbs_result.scalars().all()
    kb_perms = [{"kb_id": kb.id, "kb_name": kb.name} for kb in kbs]

    return Resp(data={"menu_permissions": menu_perms, "kb_permissions": kb_perms})


# ===== 更新角色菜单权限（不操作知识库权限）=====
class RoleMenuPermissionUpdate(BaseModel):
    permission_ids: list[int] = []

# 更新角色菜单权限（不操作知识库权限）
# ===== 角色菜单权限列表 =====
@router.put("/{role_id}/menu-permissions", response_model=Resp)
async def update_role_menu_permissions(
    role_id: int,
    body: RoleMenuPermissionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    role = await _get_role_with_perms(db, role_id)

    if role.is_admin:
        raise HTTPException(status_code=403, detail="禁止修改超级管理员角色权限")

    # 验证 permission_ids 合法
    valid_perms_result = await db.execute(select(Permission).where(Permission.id.in_(body.permission_ids)))
    valid_perms = list(valid_perms_result.scalars().all())
    if len(valid_perms) != len(body.permission_ids):
        raise HTTPException(status_code=400, detail="包含无效的权限 ID")

    # 删除旧的菜单类型角色-权限关联（保留 kb 类型）
    old_rps = await db.execute(
        select(RolePermission)
        .join(Permission, Permission.id == RolePermission.permission_id)
        .where(RolePermission.role_id == role_id, Permission.type == "menu")
    )
    for rp in old_rps.scalars().all():
        await db.delete(rp)

    # 新增菜单权限关联
    for perm in valid_perms:
        db.add(RolePermission(role_id=role_id, permission_id=perm.id))

    await db.flush()
    role = await _get_role_with_perms(db, role_id)
    await _invalidate_perm_cache()
    return Resp(data=_build_role_out(role), message="菜单权限已更新")


# ===== 更新角色权限 =====
class RolePermissionUpdate(BaseModel):
    permission_ids: list[int] = []
    kb_ids: list[int] = []  # 知识库权限

# 更新角色权限
# ===== 角色权限列表 =====
@router.put("/{role_id}/permissions", response_model=Resp)
async def update_role_permissions(
    role_id: int,
    body: RolePermissionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    role = await _get_role_with_perms(db, role_id)

    if role.is_admin:
        raise HTTPException(status_code=403, detail="禁止修改超级管理员角色权限")

    # 验证 permission_ids 合法
    valid_perms_result = await db.execute(select(Permission).where(Permission.id.in_(body.permission_ids)))
    valid_perms = list(valid_perms_result.scalars().all())
    if len(valid_perms) != len(body.permission_ids):
        raise HTTPException(status_code=400, detail="包含无效的权限 ID")

    # 验证 kb_ids 合法
    valid_kbs_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id.in_(body.kb_ids)))
    valid_kbs = list(valid_kbs_result.scalars().all())
    if len(valid_kbs) != len(body.kb_ids):
        raise HTTPException(status_code=400, detail="包含无效的知识库 ID")

    # 删除旧的菜单类型角色-权限关联（保留 kb 类型）
    old_rps = await db.execute(
        select(RolePermission)
        .join(Permission, Permission.id == RolePermission.permission_id)
        .where(RolePermission.role_id == role_id, Permission.type == "menu")
    )
    for rp in old_rps.scalars().all():
        await db.delete(rp)

    # 新增菜单权限关联
    for perm in valid_perms:
        db.add(RolePermission(role_id=role_id, permission_id=perm.id))

    # 新增知识库权限（动态创建 kb 类型权限）
    for kb_id in body.kb_ids:
        # 查找或创建 kb:{kb_id} 权限
        kb_code = f"kb:{kb_id}"
        perm_result = await db.execute(select(Permission).where(Permission.code == kb_code))
        kb_perm = perm_result.scalar_one_or_none()
        if kb_perm is None:
            kb_perm = Permission(code=kb_code, name=f"知识库{kb_id}访问权限", type="kb")
            db.add(kb_perm)
            await db.flush()
        db.add(RolePermission(role_id=role_id, permission_id=kb_perm.id))

    await db.flush()
    role = await _get_role_with_perms(db, role_id)
    await _invalidate_perm_cache()
    return Resp(data=_build_role_out(role, body.kb_ids), message="角色权限已更新")


# ===== 独立：更新角色知识库权限（不操作菜单权限）=====
class RoleKbPermissionUpdate(BaseModel):
    kb_ids: list[int]

# 更新角色知识库权限（不操作菜单权限）
# ===== 角色知识库权限列表 =====
@router.put("/{role_id}/kb-permissions", response_model=Resp)
async def update_role_kb_permissions(
    role_id: int,
    body: RoleKbPermissionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission(PermissionCode.ROLE_MANAGE)),
):
    role = await _get_role_with_perms(db, role_id)

    if role.is_admin:
        raise HTTPException(status_code=403, detail="禁止修改超级管理员角色权限")

    # 验证 kb_ids 合法
    if body.kb_ids:
        valid_kbs_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(body.kb_ids))
        )
        valid_kbs = list(valid_kbs_result.scalars().all())
        if len(valid_kbs) != len(body.kb_ids):
            raise HTTPException(status_code=400, detail="包含无效的知识库 ID")

    # 删除该角色所有 kb 类型的角色-权限关联
    old_rps = await db.execute(
        select(RolePermission)
        .join(Permission, Permission.id == RolePermission.permission_id)
        .where(RolePermission.role_id == role_id, Permission.type == "kb")
    )
    for rp in old_rps.scalars().all():
        await db.delete(rp)

    # 新增知识库权限（动态创建 kb 类型权限）
    for kb_id in body.kb_ids:
        kb_code = f"kb:{kb_id}"
        perm_result = await db.execute(select(Permission).where(Permission.code == kb_code))
        kb_perm = perm_result.scalar_one_or_none()
        if kb_perm is None:
            kb_perm = Permission(code=kb_code, name=f"知识库{kb_id}访问权限", type="kb")
            db.add(kb_perm)
            await db.flush()
        db.add(RolePermission(role_id=role_id, permission_id=kb_perm.id))

    await db.flush()
    role = await _get_role_with_perms(db, role_id)
    await _invalidate_perm_cache()
    return Resp(data=_build_role_out(role, body.kb_ids), message="知识库权限已更新")
