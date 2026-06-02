import time
from typing import List, Tuple
from app.core.config import settings
from app.core.logger import logger

try:
    from FlagEmbedding import FlagReranker
    _FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FlagReranker = None
    _FLAG_EMBEDDING_AVAILABLE = False

_cached_reranker = None
_cached_reranker_model: str = ""


def _get_reranker():
    global _cached_reranker, _cached_reranker_model
    if not _FLAG_EMBEDDING_AVAILABLE:
        raise RuntimeError("FlagEmbedding 未安装，请执行: pip install FlagEmbedding")
    model_name = settings.RERANK_MODEL
    if _cached_reranker is None or _cached_reranker_model != model_name:
        logger.info(f"Reranker 模型首次加载: {model_name}")
        # 优先尝试 local_files_only，模型已在缓存时直接加载
        try:
            _cached_reranker = FlagReranker(model_name, use_fp16=False, local_files_only=True)
            logger.info("Reranker 模型从本地缓存加载完成")
        except Exception:
            logger.info("Reranker 本地缓存未命中，尝试从 Hub 下载...")
            _cached_reranker = FlagReranker(model_name, use_fp16=False)
            logger.info("Reranker 模型下载并加载完成")
        _cached_reranker_model = model_name
    return _cached_reranker


class RerankerService:

    @classmethod
    def preload(cls):
        _get_reranker()
        logger.info(f"Reranker 模型 {settings.RERANK_MODEL} 预加载完成")

    @classmethod
    def rerank(
        cls,
        query: str,
        candidates: List[Tuple[str, str]],
    ) -> List[float]:
        """
        对候选文档列表进行 Cross-Encoder 重排序。

        Args:
            query: 用户查询
            candidates: [(chroma_id, doc_text), ...]

        Returns:
            与 candidates 等长的相关性分数列表
        """
        if not candidates:
            return []

        reranker = _get_reranker()
        pairs = [[query, text] for _, text in candidates]
        t0 = time.perf_counter()
        logits = reranker.compute_score(pairs, batch_size=8)
        elapsed = (time.perf_counter() - t0) * 1000

        if isinstance(logits, float):
            logits = [logits]
        _min_l = min(logits)
        _max_l = max(logits)
        _range = _max_l - _min_l
        if _range > 1e-8:
            scores = [(s - _min_l) / _range for s in logits]
        else:
            scores = [1.0 for _ in logits]
        logger.debug(f"Reranker 精排: {elapsed:.1f}ms | candidates={len(candidates)} | "
                     f"scores={[round(s, 4) for s in scores]}")
        return scores
