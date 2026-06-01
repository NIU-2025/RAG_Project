import re
import json
import time
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.db.session import get_db
from app.models.db import User, ChatSession, ChatMessage
from app.models.schemas import ChatSessionCreate, ChatSessionOut, ChatRequest, ChatMessageOut, Resp, PageResp
from app.api.deps import get_current_user, require_permission
from app.services.llm import LLMService
from app.services.retrieval import RetrievalService
from app.services.query_rewriter import rewrite_query_if_needed
from app.core.config import settings
from app.core.logger import logger
def _filter_cited_sources(answer: str, sources: list) -> list:
    """只保留回答中实际引用的来源（匹配 [来源1]、[1] 等格式）"""
    if not sources:
        return sources
    cited_indices = set()
    # 匹配 [来源1] [来源2] 或 [1] [2] 格式
    for m in re.finditer(r"\[来源\s*(\d+)\]|\[(\d+)\]", answer):
        idx = int(m.group(1) or m.group(2))
        cited_indices.add(idx)
    # 未找到引用标记，说明 LLM 未使用任何来源，返回空列表
    if not cited_indices:
        return []
    # sources 列表下标从 0 开始，引用编号从 1 开始
    return [s for i, s in enumerate(sources, 1) if i in cited_indices]

router = APIRouter(prefix="/chat", tags=["对话"])


@router.post("/sessions", response_model=Resp)
async def create_session(
    body: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("chat")),
):
    session = ChatSession(
        user_id=current_user.id,
        kb_id=body.kb_id,
        title=body.title or "新对话",
        llm_provider=body.llm_provider,
        llm_model=body.llm_model,
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return Resp(data=ChatSessionOut.model_validate(session))


@router.get("/sessions", response_model=PageResp)
async def list_sessions(
    page: int = 1,
    page_size: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("chat")),
):
    from sqlalchemy import func
    if current_user.is_admin:
        q = select(ChatSession)
    else:
        q = select(ChatSession).where(ChatSession.user_id == current_user.id)
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    result = await db.execute(
        q.offset((page - 1) * page_size).limit(page_size).order_by(ChatSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    return PageResp(data=[ChatSessionOut.model_validate(s) for s in sessions], total=total, page=page, page_size=page_size)


@router.get("/sessions/{session_id}/messages", response_model=Resp)
async def get_messages(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("chat")),
):
    session = await _get_session_or_404(session_id, current_user, db)
    result = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    return Resp(data=[ChatMessageOut.model_validate(m) for m in messages])


@router.delete("/sessions/{session_id}", response_model=Resp)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("chat")),
):
    session = await _get_session_or_404(session_id, current_user, db)
    await db.delete(session)
    return Resp(message="对话已删除")


@router.post("/completions")
async def chat_completions(
    body: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_permission("chat")),
):
    """RAG 问答，支持流式和非流式"""
    t_req = time.perf_counter()
    logger.info(f"Chat请求开始 | user_id={current_user.id} | session_id={body.session_id} | stream={body.stream} | msg_len={len(body.message)}")

    t_session = time.perf_counter()
    session = await _get_session_or_404(body.session_id, current_user, db)
    logger.debug(f"Chat阶段1-会话查询: {(time.perf_counter()-t_session)*1000:.1f}ms")

    # 获取历史消息（最近10轮）
    t_hist = time.perf_counter()
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == body.session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(20)
    )
    history = list(reversed(result.scalars().all()))
    logger.debug(f"Chat阶段2-历史消息查询: {(time.perf_counter()-t_hist)*1000:.1f}ms | history_len={len(history)}")

    # 保存用户消息
    t_save = time.perf_counter()
    user_msg = ChatMessage(session_id=body.session_id, role="user", content=body.message)
    db.add(user_msg)
    await db.flush()
    logger.debug(f"Chat阶段3-保存用户消息: {(time.perf_counter()-t_save)*1000:.1f}ms | user_msg_id={user_msg.id}")

    # Query 改写（多轮对话中按需触发规则/LLM改写，仅用于检索）
    t_rewrite = time.perf_counter()
    rewrite_history = [{"role": m.role, "content": m.content} for m in history]
    search_query = await rewrite_query_if_needed(
        current_query=body.message,
        history=rewrite_history,
        provider=session.llm_provider,
        db=db,
    )
    rewrite_ms = (time.perf_counter() - t_rewrite) * 1000
    if search_query != body.message:
        logger.info(f"Chat阶段3.5-Query改写: {rewrite_ms:.1f}ms | 原文=[{body.message[:30]}] | 改写=[{search_query[:50]}]")
    else:
        logger.debug(f"Chat阶段3.5-Query改写: 无需改写 ({rewrite_ms:.1f}ms)")

    # 检索相关文档（用改写后的 query）
    sources = []
    context = ""
    t_retrieval = time.perf_counter()
    if session.kb_id:
        t0 = time.perf_counter()
        search_results = await RetrievalService.search(
            kb_id=session.kb_id,
            query=search_query,
            top_k=settings.RETRIEVAL_TOP_K,
            score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
            db=db,
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Chat阶段4-检索: {retrieval_ms:.1f}ms | top_k={settings.RETRIEVAL_TOP_K} | 命中={len(search_results)} 条")
        if search_results:
            context_parts = []
            for i, r in enumerate(search_results, 1):
                context_parts.append(f"[{i}] 来源：{r.filename}（第{r.page_num or '-'}页）\n{r.content}")
                src = r.model_dump()  # 转换为字典，方便后续处理
                src["citation_index"] = i  # 记录原始编号，用于回答后精准匹配
                sources.append(src)
            context = "\n\n".join(context_parts)  # 合并所有文档内容，用双换行隔开
    else:
        logger.debug("Chat阶段4-检索: 跳过（kb_id=None）")

    # 构建对话历史
    messages = []
    for msg in history[-10:]:
        messages.append({"role": msg.role, "content": msg.content})

    if body.stream:
        logger.info(f"Chat阶段5-开始LLM流式调用 | provider={session.llm_provider} | model={session.llm_model} | context_len={len(context)}")
        return StreamingResponse(
            _stream_response(body, session, context, sources, messages, db),
            media_type="text/event-stream",
        )
    else:
        t_llm = time.perf_counter()
        answer, _ = await LLMService.chat(
            provider=session.llm_provider,
            model=session.llm_model,
            messages=messages,
            user_message=body.message,
            context=context,
            db=db,
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000
        logger.info(f"Chat阶段5-LLM调用(非流式): {llm_ms:.1f}ms | provider={session.llm_provider} | model={session.llm_model} | 回答长度={len(answer)}字")

        # 过滤：只保留回答中实际引用的来源
        t_filter = time.perf_counter()
        cited_sources = _filter_cited_sources(answer, sources)
        logger.debug(f"Chat阶段6-过滤引用来源: {(time.perf_counter()-t_filter)*1000:.1f}ms | 原始来源={len(sources)} | 引用来源={len(cited_sources)}")

        sources_json = json.dumps(cited_sources, ensure_ascii=False) if cited_sources else None
        assistant_msg = ChatMessage(
            session_id=body.session_id,
            role="assistant",
            content=answer,
            sources=sources_json,
        )
        db.add(assistant_msg)
        await db.flush()

        total_ms = (time.perf_counter() - t_req) * 1000
        logger.info(f"Chat请求完成 | 总耗时={total_ms:.1f}ms | session_id={body.session_id}")

        return Resp(data={
            "answer": answer,
            "sources": cited_sources,
            "message_id": assistant_msg.id,
        })


async def _stream_response(body, session, context, sources, messages, db):
    """流式生成器"""
    full_answer = ""
    t_llm = time.perf_counter()
    t_db_save = None
    try:
        async for chunk in LLMService.chat_stream(
            provider=session.llm_provider,
            model=session.llm_model,
            messages=messages,
            user_message=body.message,
            context=context,
            db=db,
        ):
            full_answer += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

        llm_ms = (time.perf_counter() - t_llm) * 1000
        logger.info(f"Chat阶段5-LLM流式生成完成: {llm_ms:.1f}ms | provider={session.llm_provider} | model={session.llm_model} | 回答长度={len(full_answer)}字")

        # 回答生成完毕后，过滤只保留实际引用的来源
        t_filter = time.perf_counter()
        cited_sources = _filter_cited_sources(full_answer, sources)
        logger.debug(f"Chat阶段6-过滤引用来源: {(time.perf_counter()-t_filter)*1000:.1f}ms | 原始来源={len(sources)} | 引用来源={len(cited_sources)}")
        yield f"data: {json.dumps({'type': 'sources', 'sources': cited_sources}, ensure_ascii=False)}\n\n"

        # 保存助手消息
        t_db_save = time.perf_counter()
        sources_json = json.dumps(cited_sources, ensure_ascii=False) if cited_sources else None
        assistant_msg = ChatMessage(
            session_id=body.session_id,
            role="assistant",
            content=full_answer,
            sources=sources_json,
        )
        db.add(assistant_msg)
        await db.commit()
        logger.debug(f"Chat阶段7-保存助手消息并提交: {(time.perf_counter()-t_db_save)*1000:.1f}ms | assistant_msg_id={assistant_msg.id}")

        yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg.id}, ensure_ascii=False)}\n\n"
        logger.info(f"Chat流式请求完成 | 总耗时={(time.perf_counter()-t_llm)*1000:.1f}ms")
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


async def _get_session_or_404(session_id: int, user: User, db: AsyncSession) -> ChatSession:
    if user.is_admin:
        result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    else:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="对话不存在")
    return session
