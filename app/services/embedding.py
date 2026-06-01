import gc
import time
from typing import List
from app.core.config import settings
from app.core.logger import logger

_EMBEDDING_DEVICE: str = "cpu"
_EMBEDDING_IS_GPU: bool = False

# 模块级模型缓存（每个 worker 进程只加载一次，避免每次请求 8s+ 的加载开销）
_cached_model = None
_cached_model_name: str = ""


def _get_local_model():
    """单例：全局缓存 SentenceTransformer 模型，优先从本地缓存加载"""
    global _cached_model, _cached_model_name
    if _cached_model is None or _cached_model_name != settings.EMBEDDING_MODEL:
        import os

        _patch_torch_version_check()

        from sentence_transformers import SentenceTransformer
        from huggingface_hub import try_to_load_from_cache

        model_name = settings.EMBEDDING_MODEL
        logger.info(f"Embedding 模型首次加载: {model_name}（耗时较长，后续请求直接复用）")

        # 用 huggingface_hub 精确定位本地缓存目录（含完整 tokenizer + model 文件）
        config_path = try_to_load_from_cache(model_name, "config.json")
        if config_path and os.path.isfile(config_path):
            local_path = os.path.dirname(config_path)
            logger.info(f"Embedding 本地缓存路径: {local_path}")
            _cached_model = SentenceTransformer(local_path, device="cpu", local_files_only=True)
        else:
            logger.warning("Embedding 未找到本地缓存，尝试从远端下载...")
            _cached_model = SentenceTransformer(model_name, device="cpu")

        _cached_model_name = model_name
        logger.info(f"Embedding 模型加载完成，已缓存")
    return _cached_model


def _patch_torch_version_check():
    """绕过 transformers 4.57+ 对 torch>=2.6 的硬校验"""
    try:
        import torch
        from packaging import version as _version
        if _version.parse(torch.__version__) >= _version.parse("2.6"):
            return
        import transformers.modeling_utils
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
        logger.debug("Embedding: torch 版本校验已绕过")
    except Exception:
        pass


class EmbeddingService:
    @classmethod
    def embed_texts(cls, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if settings.EMBEDDING_PROVIDER == "openai":
            return cls._openai_embed(texts)
        return cls._local_embed(texts)

    @classmethod
    def embed_query(cls, query: str) -> List[float]:
        return cls.embed_texts([query])[0]

    @classmethod
    def preload_model(cls):
        """供 main.py 启动预热调用（与 _get_local_model 共用同一缓存）"""
        _get_local_model()
        logger.info(f"Embedding 模型 {settings.EMBEDDING_MODEL} 预加载完成")

    @classmethod
    def _local_embed(cls, texts: List[str]) -> List[List[float]]:
        """使用缓存的模型编码文本块。CPU 每批 4 条避免内存爆炸。"""
        t_load = time.perf_counter()
        model = _get_local_model()
        load_ms = (time.perf_counter() - t_load) * 1000
        # 首次加载超过 100ms 说明是真实加载，否则是复用缓存
        is_first_load = load_ms > 100
        if is_first_load:
            logger.info(f"Embedding 模型加载(冷启动): {load_ms:.1f}ms | model={settings.EMBEDDING_MODEL}")
        else:
            logger.debug(f"Embedding 模型获取(复用): {load_ms:.1f}ms | model={settings.EMBEDDING_MODEL}")

        results = []
        BATCH = 4
        t_encode_total = time.perf_counter()
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            t_batch = time.perf_counter()

            vecs = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            batch_ms = (time.perf_counter() - t_batch) * 1000
            logger.debug(f"Embedding 批次 {i//BATCH+1} encode: {batch_ms:.1f}ms | batch_size={len(batch)}")
            results.extend(vecs.tolist())
            del vecs
            gc.collect()

        encode_total_ms = (time.perf_counter() - t_encode_total) * 1000
        logger.debug(f"Embedding 总encode耗时: {encode_total_ms:.1f}ms | 总文本数={len(texts)}")
        return results

    @classmethod
    def _openai_embed(cls, texts: List[str]) -> List[List[float]]:
        from openai import OpenAI
        t_api = time.perf_counter()
        client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=texts,
        )
        api_ms = (time.perf_counter() - t_api) * 1000
        logger.debug(f"Embedding-OpenAI API调用: {api_ms:.1f}ms | texts={len(texts)}")
        return [item.embedding for item in response.data]
