from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.db.session import get_db
from app.models.db import User, KnowledgeBase, Document, DocumentStatus, ChatSession, ChatMessage
from app.api.deps import require_permission
from datetime import datetime, timedelta
from collections import defaultdict

router = APIRouter(prefix="/stats", tags=["统计"])


async def _get_all_kb_ids(db: AsyncSession) -> list[int]:
    """获取所有知识库的 ID 列表（仪表盘显示所有数据）"""
    result = await db.execute(select(KnowledgeBase.id))
    return [row[0] for row in result.fetchall()]


@router.get("/overview")
async def stats_overview(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("stats")),
):
    # 仪表盘显示所有知识库数据（不受权限过滤）
    visible_kb_ids = await _get_all_kb_ids(db)

    if not visible_kb_ids:
        return {
            "total_kbs": 0,
            "total_docs": 0,
            "total_size_bytes": 0,
            "total_size_display": "0 B",
            "total_chunks": 0,
        }

    row = await db.execute(
        select(
            func.count(Document.id),
            func.coalesce(func.sum(Document.file_size), 0),
        )
        .where(Document.kb_id.in_(visible_kb_ids))
    )
    r = row.one()

    total_chunks = await db.scalar(
        select(func.coalesce(func.sum(Document.chunk_count), 0))
        .where(Document.kb_id.in_(visible_kb_ids))
    ) or 0

    return {
        "total_kbs": len(visible_kb_ids),
        "total_docs": r[0] or 0,
        "total_size_bytes": r[1] or 0,
        "total_size_display": _format_size(r[1] or 0),
        "total_chunks": total_chunks,
    }


@router.get("/kbs")
async def stats_by_kb(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("stats")),
):
    # 仪表盘显示所有知识库数据（不受权限过滤）
    visible_kb_ids = await _get_all_kb_ids(db)
    if not visible_kb_ids:
        return []

    kbs = await db.execute(
        select(KnowledgeBase)
        .where(KnowledgeBase.id.in_(visible_kb_ids))
        .order_by(KnowledgeBase.created_at.desc())
    )
    kbs = kbs.scalars().all()

    result = []
    for kb in kbs:
        docs_q = await db.execute(select(Document).where(Document.kb_id == kb.id))
        docs = docs_q.scalars().all()

        total_size = sum(d.file_size or 0 for d in docs)
        status_map = defaultdict(int)
        type_map: dict = defaultdict(lambda: {"count": 0, "size": 0})

        for d in docs:
            status_map[d.status.value] += 1
            type_map[d.file_type]["count"] += 1
            type_map[d.file_type]["size"] += d.file_size or 0

        result.append({
            "kb_id": kb.id,
            "kb_name": kb.name,
            "doc_count": len(docs),
            "total_size_bytes": total_size,
            "total_size_display": _format_size(total_size),
            "success_count": status_map.get("completed", 0),
            "failed_count": status_map.get("failed", 0),
            "pending_count": status_map.get("pending", 0),
            "file_type_breakdown": {
                ft: {
                    "count": v["count"],
                    "size": v["size"],
                    "size_display": _format_size(v["size"]),
                }
                for ft, v in type_map.items()
            },
        })
    return result


@router.get("/parse")
async def stats_parse(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("stats")),
):
    # 仪表盘显示所有知识库数据（不受权限过滤）
    visible_kb_ids = await _get_all_kb_ids(db)
    if not visible_kb_ids:
        return {
            "total": 0, "success": 0, "failed": 0, "pending": 0,
            "success_rate": 0.0, "failed_rate": 0.0,
        }

    total = await db.scalar(
        select(func.count()).select_from(Document).where(Document.kb_id.in_(visible_kb_ids))
    ) or 0
    success = await db.scalar(
        select(func.count()).select_from(Document)
        .where(Document.kb_id.in_(visible_kb_ids))
        .where(Document.status == DocumentStatus.completed)
    ) or 0
    failed = await db.scalar(
        select(func.count()).select_from(Document)
        .where(Document.kb_id.in_(visible_kb_ids))
        .where(Document.status == DocumentStatus.failed)
    ) or 0
    pending = await db.scalar(
        select(func.count()).select_from(Document)
        .where(Document.kb_id.in_(visible_kb_ids))
        .where(Document.status == DocumentStatus.pending)
    ) or 0

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "pending": pending,
        "success_rate": round(success / total, 4) if total else 0,
        "failed_rate": round(failed / total, 4) if total else 0,
    }


@router.get("/chat/daily")
async def stats_chat_daily(
    days: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("stats")),
):
    since = datetime.now() - timedelta(days=days)

    rows = await db.execute(
        select(
            func.date(ChatSession.created_at).label("date"),
            func.count(func.distinct(ChatSession.id)).label("sessions"),
            func.count(ChatMessage.id).label("messages"),
        )
        .join(ChatMessage, ChatMessage.session_id == ChatSession.id, isouter=True)
        .where(ChatSession.created_at >= since)
        .group_by(func.date(ChatSession.created_at))
        .order_by(func.date(ChatSession.created_at))
    )

    return [
        {
            "date": str(r.date),
            "sessions": r.sessions or 0,
            "messages": r.messages or 0,
        }
        for r in rows.all()
    ]


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.1f} GB"
