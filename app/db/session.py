from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine, select, text
from app.core.config import settings
from app.models.db import Base

# 异步引擎（日常使用）
async_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=(settings.APP_ENV == "development"),
)

#SQLAlchemy 异步数据库会话的配置，用于创建数据库连接工厂。
#class_=AsyncSession 指定会话类型为异步会话类
#expire_on_commit=False 提交事务后不自动过期对象（保持对象可访问）
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# 同步引擎（Alembic 迁移用）
sync_engine = create_engine(settings.DATABASE_URL_SYNC, echo=False)

# 异步数据库会话管理器
# 用于在异步请求中获取数据库会话
# 会话在请求处理完成后自动提交或回滚
async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """初始化数据库表，并执行必要的列迁移"""
    async with async_engine.begin() as conn:
        #conn.run_sync() 是 SQLAlchemy 异步引擎提供的一个方法，用于 在异步上下文中执行同步代码 。
        
        '''
        Base.metadata.create_all 的作用：
        - 扫描所有继承自 Base 的 SQLAlchemy 模型类
        - 根据模型定义自动创建对应的数据库表
        - 如果表已经存在，则 不会重复创建 （安全操作）
        '''
        await conn.run_sync(Base.metadata.create_all)

    # 初始化 admin 超级管理员账号
    await _init_admin_user()


async def _init_admin_user():
    """若 admin 用户不存在则自动创建，并初始化权限体系"""
    from app.models.db import User, Role, Permission, PermissionCode
    from app.core.security import get_password_hash
    from sqlalchemy.exc import IntegrityError

    async with AsyncSessionLocal() as session:
        # 检查 admin 用户是否已存在
        result = await session.execute(select(User).where(User.username == "admin"))
        admin = result.scalar_one_or_none()

        # 初始化权限数据
        await _init_permissions(session)

        # 创建 admin 角色（超级管理员）
        await _init_admin_role(session)

        if not admin:
            admin = User(
                username="admin",
                email="admin@rag-system.local",
                hashed_password=get_password_hash("admin123"),
                is_active=True,
            )
            session.add(admin)
            await session.flush()
            # 自动分配 admin 角色给 admin 用户
            role_result = await session.execute(select(Role).where(Role.name == "超级管理员"))
            admin_role = role_result.scalar_one_or_none()
            if admin_role:
                from app.models.db import UserRole
                session.add(UserRole(user_id=admin.id, role_id=admin_role.id))
            await session.commit()
            from app.core.logger import logger
            logger.info("已自动创建 admin 账号（初始密码: admin123，请及时修改）")
        else:
            await session.commit()


# ---------- 权限初始化（幂等操作）----------
from app.models.db import Permission as PModel, Role as RModel, UserRole as URModel

PERMISSIONS_DATA = [
    {"code": "kb_manage",   "name": "知识库管理",    "type": "menu"},
    {"code": "chat",         "name": "问答对话",       "type": "menu"},
    {"code": "stats",        "name": "仪表盘统计",     "type": "menu"},
    {"code": "model_config",  "name": "模型配置",       "type": "menu"},
    {"code": "voice_config",  "name": "语音配置",       "type": "menu"},
    {"code": "user_manage",  "name": "用户管理",       "type": "menu"},
    {"code": "role_manage",  "name": "角色管理",       "type": "menu"},
]


async def _init_permissions(session: AsyncSession):
    """幂等初始化权限数据，并清理已废弃的权限"""
    for pdata in PERMISSIONS_DATA:
        result = await session.execute(
            select(PModel).where(PModel.code == pdata["code"])
        )
        if result.scalar_one_or_none() is None:
            session.add(PModel(**pdata))
    await session.flush()

    # 清理已废弃的权限（合并到 kb_manage 的细粒度权限）
    deprecated = ["kb_read", "kb_write", "kb_delete", "doc_upload", "doc_delete", "search"]
    for code in deprecated:
        result = await session.execute(select(PModel).where(PModel.code == code))
        old = result.scalar_one_or_none()
        if old:
            # 找出关联了废弃权限的角色，给它们补上 kb_manage
            from app.models.db import RolePermission
            # 找出关联了废弃权限的角色
            roles_with_old = await session.execute(
                select(RolePermission).where(RolePermission.permission_id == old.id)
            )
            for rp in roles_with_old.scalars().all():
                # 检查是否已有 kb_manage
                kb_manage_perm = await session.execute(
                    select(PModel).where(PModel.code == "kb_manage")
                )
                kbm = kb_manage_perm.scalar_one_or_none()
                if kbm:
                    # 避免重复关联 能查询到数据则说明已关联
                    dup = await session.execute(
                        select(RolePermission).where(
                            RolePermission.role_id == rp.role_id,
                            RolePermission.permission_id == kbm.id,
                        )
                    )
                    if dup.scalar_one_or_none() is None:
                        session.add(RolePermission(role_id=rp.role_id, permission_id=kbm.id))
            # 删除废弃权限记录
            await session.delete(old)
    await session.flush()


async def _init_admin_role(session: AsyncSession):
    """幂等初始化超级管理员角色，并关联所有权限"""
    result = await session.execute(select(RModel).where(RModel.name == "超级管理员"))
    role = result.scalar_one_or_none()
    if role is None:
        role = RModel(name="超级管理员", description="系统超级管理员，拥有所有权限", is_admin=True)
        session.add(role)
        await session.flush()
        # 关联所有权限
        all_perms = await session.execute(select(PModel))
        from app.models.db import RolePermission
        for perm in all_perms.scalars().all():
            session.add(RolePermission(role_id=role.id, permission_id=perm.id))
    await session.flush()
