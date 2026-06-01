from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.vector_store import VectorStore
from app.models.db import DocumentChunk, Document
from app.models.schemas import SearchResult
from app.services.embedding import EmbeddingService
from app.core.config import settings
from app.core.logger import logger
import jieba
import time


class RetrievalService:

    @staticmethod
    async def search(
        kb_id: int,
        query: str,
        top_k: int = None,
        score_threshold: float = None,
        file_type: Optional[str] = None,
        tags: Optional[str] = None,
        db: AsyncSession = None,
    ) -> List[SearchResult]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD

        # 1. Query Embedding
        t0 = time.perf_counter()
        query_embedding = EmbeddingService.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"检索-Query embedding: {embed_ms:.1f}ms")
        return await RetrievalService._search_with_embedding(
            kb_id=kb_id,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            file_type=file_type,
            tags=tags,
            db=db,
        )

    @staticmethod
    async def search_with_embedding(
        kb_id: int,
        query: str,
        query_embedding: List[float],
        top_k: int = None,
        score_threshold: float = None,
        file_type: Optional[str] = None,
        tags: Optional[str] = None,
        db: AsyncSession = None,
    ) -> List[SearchResult]:
        top_k = top_k or settings.RETRIEVAL_TOP_K
        score_threshold = score_threshold if score_threshold is not None else settings.RETRIEVAL_SCORE_THRESHOLD
        return await RetrievalService._search_with_embedding(
            kb_id=kb_id,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            file_type=file_type,
            tags=tags,
            db=db,
        )

    @staticmethod
    async def _search_with_embedding(
        kb_id: int,
        query: str,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        file_type: Optional[str] = None,
        tags: Optional[str] = None,
        db: AsyncSession = None,
    ) -> List[SearchResult]:

        # 2. ChromaDB 向量检索
        t1 = time.perf_counter()
        where_filter = _build_where(kb_id, file_type, tags)
        vector_results = VectorStore.query(
            kb_id=kb_id,
            query_embedding=query_embedding,
            top_k=top_k * 2,
            where=where_filter,
        )
        chroma_ms = (time.perf_counter() - t1) * 1000
        vec_count = len(vector_results.get("ids", [[]])[0]) if vector_results else 0
        logger.debug(f"检索-ChromaDB查询: {chroma_ms:.1f}ms | 命中向量={vec_count} 条")

        # 3. 解析向量检索结果
        t_parse = time.perf_counter()
        vec_scores: Dict[str, float] = {}
        vec_docs: Dict[str, dict] = {}

        if vector_results["ids"] and vector_results["ids"][0]:
            for chroma_id, distance, doc_text, metadata in zip(
                vector_results["ids"][0],
                vector_results["distances"][0],
                vector_results["documents"][0],
                vector_results["metadatas"][0],
            ):
                similarity = 1.0 - distance
                vec_scores[chroma_id] = similarity
                vec_docs[chroma_id] = {"text": doc_text, "metadata": metadata}
        parse_ms = (time.perf_counter() - t_parse) * 1000
        logger.debug(f"检索-解析向量结果: {parse_ms:.1f}ms | vec_scores={len(vec_scores)}")

        # 4. BM25 检索
        t_bm25 = time.perf_counter()
        bm25_scores = await _bm25_search(kb_id, query, top_k * 2, db)
        bm25_ms = (time.perf_counter() - t_bm25) * 1000
        logger.debug(f"检索-BM25检索: {bm25_ms:.1f}ms | bm25命中={len(bm25_scores)} 条")

        # 5. 合并所有候选
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())

        # 6. 计算加权混合分数，按此排序取 Top-K
        t_fusion = time.perf_counter()

        # 动态权重：某一方无结果时，另一方权重升至 1.0
        if len(vec_scores) == 0 and len(bm25_scores) > 0:
            vec_w, bm25_w = 0.0, 1.0
        elif len(bm25_scores) == 0 and len(vec_scores) > 0:
            vec_w, bm25_w = 1.0, 0.0
        else:
            vec_w, bm25_w = settings.VECTOR_WEIGHT, settings.BM25_WEIGHT

        combined_scores: Dict[str, float] = {}
        for chroma_id in all_ids:
            vec_s = vec_scores.get(chroma_id, 0.0)
            bm25_s = bm25_scores.get(chroma_id, 0.0)
            combined_scores[chroma_id] = vec_s * vec_w + bm25_s * bm25_w

        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        fusion_ms = (time.perf_counter() - t_fusion) * 1000
        logger.debug(f"检索-分数融合: {fusion_ms:.1f}ms | 候选数={len(all_ids)} | vec_weight={vec_w} | bm25_weight={bm25_w}")

        # 7. 收集候选文档（含纯 BM25 命中的 fallback DB 补全）
        rerank_multiplier = settings.RERANK_MULTIPLIER if settings.RERANK_ENABLED else 1
        coarse_top_n = min(top_k * rerank_multiplier, len(sorted_ids))
        coarse_ids = sorted_ids[:coarse_top_n]

        missing_ids = [cid for cid in coarse_ids if cid not in vec_docs]
        fallback_map = await _batch_lookup_chunks(missing_ids, db) if missing_ids and db else {}

        # 8. Cross-Encoder 重排序（精排阶段）
        t_rerank = time.perf_counter()
        if settings.RERANK_ENABLED and coarse_ids and len(coarse_ids) > 1:
            from app.services.reranker import RerankerService

            candidates = []
            for cid in coarse_ids:
                info = vec_docs.get(cid) or fallback_map.get(cid)
                if info:
                    candidates.append((cid, info["text"]))

            rerank_scores = RerankerService.rerank(query, candidates)

            rerank_map: Dict[str, float] = {}
            for (cand_id, _), rs in zip(candidates, rerank_scores):
                rerank_map[cand_id] = rs

            final_scores = rerank_map
            final_sorted = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:top_k]
            rerank_ms = (time.perf_counter() - t_rerank) * 1000
            logger.info(f"检索-Reranker精排: {rerank_ms:.1f}ms | 粗排候选={len(coarse_ids)} | 精排后={len(final_sorted)}")
        else:
            final_scores = combined_scores
            final_sorted = coarse_ids[:top_k]
            if settings.RERANK_ENABLED and coarse_ids:
                logger.debug(f"检索-Reranker跳过: 候选数={len(coarse_ids)}（<=1 条无需重排）")

        # 9. 构建返回结果
        # Reranker 精排路径：使用相对排序而非绝对阈值截断
        # 只要 top-1 得分 > 0.01（并非完全不相关），就取 top_k 返回
        use_reranked = settings.RERANK_ENABLED and len(coarse_ids) > 1
        results = []
        for chroma_id in final_sorted[:top_k]:
            score = final_scores.get(chroma_id, combined_scores.get(chroma_id, 0.0))

            if use_reranked:
                top_score = final_scores.get(final_sorted[0], 0) if final_sorted else 0
                if top_score < 0.01:
                    logger.debug(f"检索-Reranker判定全部不相关: top_score={top_score:.4f}")
                    break
            else:
                if score < score_threshold:
                    logger.debug(f"检索-分数过滤: chroma_id={chroma_id} score={score:.4f} < threshold={score_threshold}")
                    continue

            info = vec_docs.get(chroma_id) or fallback_map.get(chroma_id)
            if not info:
                continue

            meta = info["metadata"]
            results.append(SearchResult(
                doc_id=meta.get("doc_id", 0),
                filename=meta.get("filename", ""),
                file_type=meta.get("file_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                page_num=meta.get("page_num") or None,
                content=info["text"],
                score=round(score, 4),
                tags=meta.get("tags") or None,
            ))

        logger.debug(f"检索完成: kb={kb_id}, query={query[:30]}, 命中={len(results)}")
        return results


async def _bm25_search(
    kb_id: int, query: str, top_k: int, db: AsyncSession
) -> Dict[str, float]:
    """基于 BM25 的关键词检索（Redis 缓存分词语料加速）"""
    if not db:
        return {}

    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus, chroma_ids = await _get_bm25_corpus(kb_id, db)
        if not tokenized_corpus or len(tokenized_corpus) <= 2:
            return {}

        query_tokens = list(jieba.cut(query))
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)

        max_score = max(scores) if max(scores) > 0 else 1.0
        bm25_result = {}
        for i, cid in enumerate(chroma_ids):
            if scores[i] > 0:
                bm25_result[cid] = scores[i] / max_score

        sorted_ids = sorted(bm25_result.keys(), key=lambda x: bm25_result[x], reverse=True)[:top_k]
        return {k: bm25_result[k] for k in sorted_ids}

    except Exception as e:
        logger.warning(f"BM25 检索失败，降级为纯向量检索: {e}")
        return {}


async def _get_bm25_corpus(kb_id: int, db: AsyncSession):
    """获取 BM25 语料（优先从 Redis 缓存读取）"""
    from app.core.redis_client import cache_get_json, cache_set_json

    cache_key = f"kb:{kb_id}:bm25"

    cached = await cache_get_json(cache_key)
    if cached and "ids" in cached and "corpus" in cached:
        logger.debug(f"BM25 缓存命中: kb_id={kb_id}, chunks={len(cached['ids'])}")
        return cached["corpus"], cached["ids"]

    t_db = time.perf_counter()
    result = await db.execute(
        select(DocumentChunk.chroma_id, DocumentChunk.content)
        .where(DocumentChunk.kb_id == kb_id)
    )
    rows = result.all()
    db_ms = (time.perf_counter() - t_db) * 1000

    if not rows or len(rows) <= 2:
        return [], []

    t_jieba = time.perf_counter()
    chroma_ids = [row.chroma_id for row in rows]
    tokenized_corpus = [list(jieba.cut(row.content)) for row in rows]
    jieba_ms = (time.perf_counter() - t_jieba) * 1000
    logger.info(f"BM25 缓存未命中，重新构建: db={db_ms:.1f}ms jieba={jieba_ms:.1f}ms chunks={len(rows)}")

    await cache_set_json(cache_key, {"ids": chroma_ids, "corpus": tokenized_corpus}, ttl=settings.BM25_CACHE_TTL)

    return tokenized_corpus, chroma_ids


async def _batch_lookup_chunks(
    chroma_ids: List[str], db: AsyncSession
) -> Dict[str, dict]:
    """批量查询 BM25-only 结果的 chunk 元数据（含文档名/类型）"""
    if not chroma_ids or db is None:
        return {}
    try:
        result = await db.execute(
            select(
                DocumentChunk.chroma_id,
                DocumentChunk.content,
                DocumentChunk.doc_id,
                DocumentChunk.chunk_index,
                DocumentChunk.page_num,
                Document.filename,
                Document.file_type,
                Document.tags,
            )
            .join(Document, DocumentChunk.doc_id == Document.id)
            .where(DocumentChunk.chroma_id.in_(chroma_ids))
        )
        rows = result.all()
        fallback = {}
        for row in rows:
            fallback[row.chroma_id] = {
                "text": row.content,
                "metadata": {
                    "doc_id": row.doc_id,
                    "filename": row.filename or "",
                    "file_type": row.file_type or "",
                    "chunk_index": row.chunk_index or 0,
                    "page_num": row.page_num,
                    "tags": row.tags or "",
                },
            }
        logger.debug(f"BM25 fallback 批量查DB: {len(chroma_ids)} IDs → {len(fallback)} 命中")
        return fallback
    except Exception as e:
        logger.warning(f"BM25 fallback DB查询失败: {e}")
        return {}


def _build_where(kb_id: int, file_type: Optional[str], tags: Optional[str]) -> Optional[Dict]:
    conditions = []
    if file_type:
        conditions.append({"file_type": {"$eq": file_type}})
    if tags:
        conditions.append({"tags": {"$contains": tags}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
