import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import asyncio
from app.db.session import get_db
from app.models.db import User, KnowledgeBase, Document, DocumentStatus
from app.models.schemas import DocumentOut, Resp, PageResp
from app.api.deps import get_current_user, get_current_user_optional, require_permission
from app.core.config import settings
from app.core.logger import logger
from app.services.document import DocumentService
from typing import Optional

router = APIRouter(prefix="/docs", tags=["文档管理"])

ALLOWED_TYPES = {"pdf", "docx", "xlsx", "xls", "txt", "md", "csv"}
ALLOWED_IMAGE_TYPES = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
ALLOWED_AUDIO_TYPES = {"mp3", "wav", "m4a", "aac", "ogg", "wma", "flac", "pcm"}


@router.post("/upload", response_model=Resp)
async def upload_document(
    kb_id: int = Form(..., description="知识库ID"),
    tags: Optional[str] = Form(None, description="文档标签，逗号分隔"),
    file: UploadFile = File(..., description="文档文件"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    # 验证知识库权限：所有者、或公开、或角色授权
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if kb.owner_id != current_user.id and not current_user.is_admin:
        from app.api.deps import get_user_kb_ids
        user_kb_ids = await get_user_kb_ids(db, current_user)
        if kb.id not in user_kb_ids:
            raise HTTPException(status_code=403, detail="无权向该知识库上传文档")

    # 同知识库内不允许重复文件名
    existing = await db.execute(
        select(Document).where(Document.kb_id == kb_id, Document.filename == file.filename)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"知识库内已存在同名文件「{file.filename}」")

    # 校验文件类型
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    is_image = ext in ALLOWED_IMAGE_TYPES
    is_audio = ext in ALLOWED_AUDIO_TYPES
    allowed = ALLOWED_IMAGE_TYPES if is_image else ALLOWED_TYPES
    if is_audio:
        allowed = ALLOWED_AUDIO_TYPES
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 读取文件内容
    content = await file.read()
    content_size = len(content)

    # 校验文件大小（图片/语音使用各自的限制）
    if is_audio:
        if content_size > settings.MAX_AUDIO_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"音频文件超过 {settings.MAX_AUDIO_SIZE_MB}MB 限制")
    elif is_image:
        if content_size > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"图片超过 {settings.MAX_IMAGE_SIZE_MB}MB 限制")
    else:
        if content_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"文件超过 {settings.MAX_FILE_SIZE_MB}MB 限制")

    # 保存原始文件
    save_dir = os.path.join(settings.UPLOAD_DIR, str(kb_id))
    os.makedirs(save_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(save_dir, unique_name)
    with open(file_path, "wb") as f:
        f.write(content)

    # 创建文档记录，状态直接设为 processing，立即返回给前端
    doc = Document(
        kb_id=kb_id,
        filename=file.filename,
        file_path=file_path,
        file_type=ext,
        file_size=len(content),
        tags=tags,
        status=DocumentStatus.processing,
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)
    await db.commit()

    # 真正的后台异步处理：解析 + 向量化 + 入库（不阻塞请求）
    asyncio.create_task(DocumentService.process_document(doc.id))

    return Resp(data=DocumentOut.model_validate(doc), message="文件上传成功，正在后台处理")


@router.get("", response_model=PageResp)
async def list_documents(
    kb_id: int,
    page: int = 1,
    page_size: int = 20,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    from app.api.deps import get_user_kb_ids
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    kb = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if kb.owner_id != current_user.id and not current_user.is_admin:
        user_kb_ids = await get_user_kb_ids(db, current_user)
        if kb.id not in user_kb_ids:
            raise HTTPException(status_code=403, detail="无权访问该知识库")

    q = select(Document).where(Document.kb_id == kb_id)
    if status:
        q = q.where(Document.status == status)
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    result = await db.execute(
        q.offset((page - 1) * page_size).limit(page_size).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return PageResp(data=[DocumentOut.model_validate(d) for d in docs], total=total, page=page, page_size=page_size)


@router.get("/{doc_id}", response_model=Resp)
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    doc = await _get_doc_or_404(doc_id, db)
    from app.api.deps import get_user_kb_ids
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id))
    kb = kb_result.scalar_one_or_none()
    if kb and kb.owner_id != current_user.id and not current_user.is_admin:
        user_kb_ids = await get_user_kb_ids(db, current_user)
        if kb.id not in user_kb_ids:
            raise HTTPException(status_code=403, detail="无权访问该文档")
    return Resp(data=DocumentOut.model_validate(doc))


@router.delete("/{doc_id}", response_model=Resp)
async def delete_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    doc = await _get_doc_or_404(doc_id, db)

    # 获取知识库信息用于权限校验
    kb_result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id))
    kb = kb_result.scalar_one_or_none()

    # 超级管理员可删除任意知识库的文档
    # 有该知识库权限的用户（通过角色被授予 kb:{id} 权限）也可删除
    # 仅 KB 所有者或超级管理员无权限
    if not current_user.is_admin:
        if kb.owner_id != current_user.id:
            from app.api.deps import get_user_kb_ids
            user_kb_ids = await get_user_kb_ids(db, current_user)
            if doc.kb_id not in user_kb_ids:
                raise HTTPException(status_code=403, detail="无权删除该文档")

    # 删除向量
    from app.db.vector_store import VectorStore
    VectorStore.delete_by_doc_id(doc.kb_id, doc_id)

    # 删除原始文件
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    # 更新知识库文档数
    if kb and kb.doc_count > 0:
        kb.doc_count -= 1

    await db.delete(doc)

    from app.services.document import _invalidate_bm25_cache
    await _invalidate_bm25_cache(doc.kb_id)

    return Resp(message="文档已删除")

# 重新处理失败的文档
@router.post("/{doc_id}/reprocess", response_model=Resp)
async def reprocess_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    """重新处理失败的文档"""
    doc = await _get_doc_or_404(doc_id, db)
    doc.status = DocumentStatus.processing
    doc.error_msg = None
    await db.commit()
    asyncio.create_task(DocumentService.process_document(doc_id))
    return Resp(message="已重新提交处理")


async def _get_doc_or_404(doc_id: int, db: AsyncSession) -> Document:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return doc


@router.get("/{doc_id}/file")
async def serve_doc_file(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user_optional),
):
    """直接返回原始文档/图片文件，供前端预览和下载"""
    doc = await _get_doc_or_404(doc_id, db)

    if current_user is not None:
        from app.api.deps import get_user_kb_ids
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
        )
        kb = kb_result.scalar_one_or_none()
        if kb and kb.owner_id != current_user.id and not current_user.is_admin:
            user_kb_ids = await get_user_kb_ids(db, current_user)
            if kb.id not in user_kb_ids:
                raise HTTPException(status_code=403, detail="无权查看此文档")

    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="文件不存在，请重新上传")

    import mimetypes
    mime_type, _ = mimetypes.guess_type(doc.filename)
    if mime_type is None:
        image_exts = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
        if doc.file_type.lower() in image_exts:
            mime_type = f"image/{doc.file_type.lower()}"
        else:
            mime_type = "application/octet-stream"

    from starlette.responses import FileResponse
    return FileResponse(
        path=doc.file_path,
        filename=doc.filename,
        media_type=mime_type,
    )
