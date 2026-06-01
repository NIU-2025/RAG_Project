from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db.session import get_db
from app.db.vector_store import VectorStore
from app.models.db import User, KnowledgeBase, Document
from app.models.schemas import KBCreate, KBUpdate, KBOut, Resp, PageResp
from app.api.deps import get_current_user, require_permission, get_user_kb_ids
from typing import Optional

router = APIRouter(prefix="/kb", tags=["知识库"])


@router.post("", response_model=Resp)
async def create_kb(
    body: KBCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    kb = KnowledgeBase(**body.model_dump(), owner_id=current_user.id, is_public=True)
    db.add(kb)
    await db.flush()
    await db.refresh(kb)
    return Resp(data=KBOut.model_validate(kb))


@router.get("", response_model=PageResp)
async def list_kbs(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    offset = (page - 1) * page_size

    # 超级管理员可查看所有知识库，普通用户只能看自己的 + 角色授权的
    if current_user.is_admin:
        q = select(KnowledgeBase)
    else:
        user_kb_ids = await get_user_kb_ids(db, current_user)
        q = select(KnowledgeBase).where(
            (KnowledgeBase.owner_id == current_user.id)
            | (KnowledgeBase.id.in_(user_kb_ids))
        )
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    result = await db.execute(q.offset(offset).limit(page_size).order_by(KnowledgeBase.created_at.desc()))
    kbs = result.scalars().all()

    # 实时从 documents 表查询真实文档数，避免 doc_count 字段不同步
    if kbs:
        kb_ids = [kb.id for kb in kbs]
        count_rows = await db.execute(
            select(Document.kb_id, func.count(Document.id))
            .where(Document.kb_id.in_(kb_ids))
            .group_by(Document.kb_id)
        )
        count_map = {row[0]: row[1] for row in count_rows.fetchall()}
        for kb in kbs:
            kb.doc_count = count_map.get(kb.id, 0)

    return PageResp(data=[KBOut.model_validate(kb) for kb in kbs], total=total, page=page, page_size=page_size)


@router.get("/{kb_id}", response_model=Resp)
async def get_kb(
    kb_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    kb = await _get_kb_or_404(kb_id, current_user, db)
    # 实时查询真实文档数
    real_count = await db.scalar(
        select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
    )
    kb.doc_count = real_count or 0
    return Resp(data=KBOut.model_validate(kb))


@router.put("/{kb_id}", response_model=Resp)
async def update_kb(
    kb_id: int,
    body: KBUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    kb = await _get_kb_or_404(kb_id, current_user, db, owner_only=True)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(kb, field, value)
    await db.flush()
    await db.refresh(kb)
    return Resp(data=KBOut.model_validate(kb))


@router.delete("/{kb_id}", response_model=Resp)
async def delete_kb(
    kb_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    kb = await _get_kb_or_404(kb_id, current_user, db, owner_only=True)
    VectorStore.delete_collection(kb_id)
    await db.delete(kb)

    from app.services.document import _invalidate_bm25_cache
    await _invalidate_bm25_cache(kb_id)

    return Resp(message="知识库已删除")


async def _get_kb_or_404(
    kb_id: int, user: User, db: AsyncSession, owner_only: bool = False
) -> KnowledgeBase:
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if owner_only:
        # 写操作（更新/删除）：仅所有者或超级管理员可执行
        if kb.owner_id != user.id and not user.is_admin:
            raise HTTPException(status_code=403, detail="无权操作该知识库")
    else:
        # 读操作：所有者、或角色授权、或超级管理员可访问
        if kb.owner_id != user.id and not user.is_admin:
            user_kb_ids = await get_user_kb_ids(db, user)
            if kb.id not in user_kb_ids:
                raise HTTPException(status_code=403, detail="无权访问该知识库")
    return kb
