from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.models.db import User
from app.models.schemas import SearchRequest, SearchResponse
from app.api.deps import get_current_user, require_permission
from app.services.retrieval import RetrievalService

router = APIRouter(prefix="/search", tags=["检索"])


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("kb_manage")),
):
    """测试检索接口，直接返回相关文档片段"""
    results = await RetrievalService.search(
        kb_id=body.kb_id,
        query=body.query,
        top_k=body.top_k,
        score_threshold=body.score_threshold,
        file_type=body.file_type,
        tags=body.tags,
        db=db,
    )
    return SearchResponse(query=body.query, results=results, total=len(results))
