import gc
import os
import time
from typing import List
from app.core.config import settings
from app.core.logger import logger

_EMBEDDING_DEVICE: str | None = None
_CACHED_MODEL = None
_CACHED_MODEL_NAME: str = ""


def _detect_device() -> str:
    global _EMBEDDING_DEVICE
    if _EMBEDDING_DEVICE is not None:
        return _EMBEDDING_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            logger.info(f"Embedding: CUDA 可用, 显存={vram_mb}MB")
            _EMBEDDING_DEVICE = "cuda"
            return _EMBEDDING_DEVICE
    except Exception:
        pass
    _EMBEDDING_DEVICE = "cpu"
    return _EMBEDDING_DEVICE


def _detect_batch_size() -> int:
    device = _detect_device()
    if device == "cuda":
        try:
            import torch
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            if vram_mb < 4000:
                return 8
            elif vram_mb < 8000:
                return 16
            else:
                return 32
        except Exception:
            return 8
    return 4


def _get_local_model():
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    if _CACHED_MODEL is None or _CACHED_MODEL_NAME != settings.EMBEDDING_MODEL:
        _patch_torch_version_check()

        from sentence_transformers import SentenceTransformer
        from huggingface_hub import try_to_load_from_cache

        model_name = settings.EMBEDDING_MODEL
        device = _detect_device()
        logger.info(f"Embedding 模型首次加载: {model_name} device={device}")

        config_path = try_to_load_from_cache(model_name, "config.json")
        if config_path and os.path.isfile(config_path):
            local_path = os.path.dirname(config_path)
            _CACHED_MODEL = SentenceTransformer(local_path, device=device, local_files_only=True)
        else:
            _CACHED_MODEL = SentenceTransformer(model_name, device=device)

        _CACHED_MODEL_NAME = model_name
        logger.info(f"Embedding 模型加载完成, device={device}")
    return _CACHED_MODEL


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
        t_load = time.perf_counter()
        model = _get_local_model()
        load_ms = (time.perf_counter() - t_load) * 1000
        if load_ms > 100:
            logger.info(f"Embedding 模型加载(冷启动): {load_ms:.1f}ms | model={settings.EMBEDDING_MODEL}")
        else:
            logger.debug(f"Embedding 模型获取(复用): {load_ms:.1f}ms | model={settings.EMBEDDING_MODEL}")

        results = []
        BATCH = _detect_batch_size()
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
        logger.debug(f"Embedding 总encode耗时: {encode_total_ms:.1f}ms | 总文本数={len(texts)} | batch={BATCH}")
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
