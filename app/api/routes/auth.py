from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.db.session import get_db
from app.models.db import User, Role
from app.models.schemas import UserRegister, UserLogin, UserOut, Token, Resp
from app.core.security import get_password_hash, verify_password, create_access_token
from app.api.deps import get_current_user

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post("/register", response_model=Resp)
async def register(body: UserRegister, db: AsyncSession = Depends(get_db)):
    # 检查用户名/邮箱是否已存在
    result = await db.execute(
        select(User).where((User.username == body.username) | (User.email == body.email))
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="用户名或邮箱已被注册")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=get_password_hash(body.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    await db.commit()
    return Resp(data=UserOut.model_validate(user))


@router.post("/login", response_model=Token)
async def login(body: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.username == body.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已禁用")

    token = create_access_token({"sub": str(user.id)})

    # 加载用户角色和权限（selectinload 兼容异步 session）
    result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == user.id)
    )
    user = result.unique().scalar_one()

    return Token(access_token=token, user=UserOut.model_validate(user))


@router.get("/me/permissions", response_model=Resp)
async def get_my_permissions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    获取当前用户的角色和权限信息，供前端渲染菜单和权限判断使用。
    """
    from app.api.deps import _get_user_permissions, get_user_kb_ids

    # 用 selectinload 正确加载角色和权限
    me_result = await db.execute(
        select(User)
        .options(selectinload(User.roles).selectinload(Role.permissions))
        .where(User.id == current_user.id)
    )
    current_user = me_result.unique().scalar_one()

    user_perms = await _get_user_permissions(db, current_user.id)
    user_kb_ids = await get_user_kb_ids(db, current_user)

    # 菜单权限码列表
    menu_perm_codes = [p.code for p in user_perms if p.type == "menu"]

    # 获取当前所有知识库列表（用于角色配置知识库权限时展示）
    from app.models.db import KnowledgeBase
    from sqlalchemy import select as sa_select
    kbs_result = await db.execute(sa_select(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()))
    kbs = [
        {"id": kb.id, "name": kb.name}
        for kb in kbs_result.scalars().all()
    ]

    return Resp(data={
        "user": UserOut.model_validate(current_user),
        "is_admin": current_user.is_admin,
        "roles": [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "is_admin": r.is_admin,
                "permission_ids": [p.id for p in r.permissions],
            }
            for r in current_user.roles
        ],
        "menu_permission_codes": menu_perm_codes,
        "kb_permission_ids": user_kb_ids,
        "all_kbs": kbs,
    })
