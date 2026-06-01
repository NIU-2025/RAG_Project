import time
from typing import List, Dict, Optional, AsyncGenerator
from app.core.config import settings
from app.core.logger import logger

# RAG System Prompt
RAG_SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请优先根据提供的参考资料回答用户问题。

规则：
1. 优先使用参考资料中的内容回答，确保准确可靠。
2. 有参考资料时：
   - 严格按照资料内容回答，不要添加资料中不存在的信息
   - 回答时引用来源，格式：[来源X]
   - 如果参考资料不足以支撑完整结论，请明确说明"根据现有资料无法完全回答，以下仅基于已有内容："
3. 没有参考资料时：
   - 首先说明"知识库中未找到相关内容，以下为通用回答："
   - 然后直接回答用户的问题，回答必须准确、客观、安全，不要只停留在说明上
4. 回答语言与用户提问语言保持一致
5. 回答要简洁、准确、有条理
6. ⚠️ 严禁编造：如果资料中找不到明确依据，不得猜测或杜撰任何具体数字、规则、条款
"""

NO_CONTEXT_SYSTEM_PROMPT = """你是一个专业的AI助手，请准确、有帮助地回答用户问题。"""


def _build_messages(
    messages: List[Dict],
    user_message: str,
    context: str,
) -> List[Dict]:
    system_prompt = RAG_SYSTEM_PROMPT if context else NO_CONTEXT_SYSTEM_PROMPT

    result = [{"role": "system", "content": system_prompt}]

    # 历史消息
    result.extend(messages)

    # 当前用户消息（含检索上下文）
    if context:
        content = f"参考资料：\n{context}\n\n用户问题：{user_message}"
    else:
        content = user_message

    result.append({"role": "user", "content": content})
    return result


async def _get_provider_cfg(provider: str, db=None) -> dict:
    """从 Redis 缓存或 DB 读取 Provider 配置"""
    if db is None:
        return {}

    from app.core.redis_client import cache_get_json, cache_set_json

    cache_key = f"provider_cfg:{provider}"
    cached = await cache_get_json(cache_key)
    if cached:
        logger.debug(f"LLM Provider配置缓存命中: {provider}")
        return cached

    t_db = time.perf_counter()
    try:
        from sqlalchemy import select
        from app.models.db import ModelConfig
        result = await db.execute(
            select(ModelConfig).where(
                ModelConfig.provider == provider,
                ModelConfig.is_enabled == True,
            )
        )
        cfg = result.scalar_one_or_none()
        db_ms = (time.perf_counter() - t_db) * 1000
        logger.debug(f"LLM Provider配置DB查询: {db_ms:.1f}ms | provider={provider} | found={cfg is not None}")
        if cfg is None:
            return {}

        data = {
            "api_key": cfg.api_key,
            "api_secret": cfg.api_secret,
            "base_url": cfg.base_url,
            "model_name": cfg.model_name,
        }
        await cache_set_json(cache_key, data, ttl=settings.PROVIDER_CACHE_TTL)
        return data
    except Exception:
        db_ms = (time.perf_counter() - t_db) * 1000
        logger.warning(f"LLM Provider配置DB查询失败: {db_ms:.1f}ms | provider={provider}")
        return {}


class LLMService:

    @staticmethod
    async def chat(
        provider: str,
        model: Optional[str],
        messages: List[Dict],
        user_message: str,
        context: str = "",
        db=None,
    ):
        full_messages = _build_messages(messages, user_message, context)
        cfg = await _get_provider_cfg(provider, db)

        if provider in ("openai", "deepseek"):
            return await _openai_chat(provider, model, full_messages, stream=False, cfg=cfg)
        elif provider == "dashscope":
            return await _dashscope_chat(model, full_messages, cfg=cfg)
        elif provider == "qianfan":
            return await _qianfan_chat(model, full_messages, cfg=cfg)
        elif provider == "ollama":
            return await _ollama_chat(model, full_messages, stream=False, cfg=cfg)
        elif provider == "lmstudio":
            return await _lmstudio_chat(model, full_messages, stream=False, cfg=cfg)
        else:
            raise ValueError(f"不支持的 LLM Provider: {provider}")

    @staticmethod
    async def chat_stream(
        provider: str,
        model: Optional[str],
        messages: List[Dict],
        user_message: str,
        context: str = "",
        db=None,
    ) -> AsyncGenerator[str, None]:
        full_messages = _build_messages(messages, user_message, context)
        cfg = await _get_provider_cfg(provider, db)

        if provider in ("openai", "deepseek"):
            async for chunk in _openai_chat_stream(provider, model, full_messages, cfg=cfg):
                yield chunk
        elif provider == "dashscope":
            async for chunk in _dashscope_chat_stream(model, full_messages, cfg=cfg):
                yield chunk
        elif provider == "ollama":
            async for chunk in _ollama_chat_stream(model, full_messages, cfg=cfg):
                yield chunk
        elif provider == "lmstudio":
            async for chunk in _lmstudio_chat_stream(model, full_messages, cfg=cfg):
                yield chunk
        else:
            # 非流式降级：一次性返回
            answer, _ = await LLMService.chat(provider, model, messages, user_message, context, db)
            yield answer


# ===== Provider 实现 =====

async def _openai_chat(provider: str, model: Optional[str], messages: List[Dict], stream: bool, cfg: dict = None):
    from openai import AsyncOpenAI
    cfg = cfg or {}
    if provider == "deepseek":
        api_key = cfg.get("api_key") or settings.DEEPSEEK_API_KEY
        base_url = cfg.get("base_url") or settings.DEEPSEEK_BASE_URL
        model = model or cfg.get("model_name") or settings.DEEPSEEK_MODEL
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    else:
        api_key = cfg.get("api_key") or settings.OPENAI_API_KEY
        base_url = cfg.get("base_url") or settings.OPENAI_BASE_URL
        model = model or cfg.get("model_name") or settings.OPENAI_MODEL
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    t_api = time.perf_counter()
    response = await client.chat.completions.create(model=model, messages=messages, stream=False)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-OpenAI API调用: {api_ms:.1f}ms | provider={provider} | model={model} | msg_count={len(messages)}")
    content = response.choices[0].message.content
    return content, {}


async def _openai_chat_stream(provider: str, model: Optional[str], messages: List[Dict], cfg: dict = None) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    cfg = cfg or {}
    if provider == "deepseek":
        api_key = cfg.get("api_key") or settings.DEEPSEEK_API_KEY
        base_url = cfg.get("base_url") or settings.DEEPSEEK_BASE_URL
        model = model or cfg.get("model_name") or settings.DEEPSEEK_MODEL
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    else:
        api_key = cfg.get("api_key") or settings.OPENAI_API_KEY
        base_url = cfg.get("base_url") or settings.OPENAI_BASE_URL
        model = model or cfg.get("model_name") or settings.OPENAI_MODEL
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    t_api = time.perf_counter()
    stream = await client.chat.completions.create(model=model, messages=messages, stream=True)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-OpenAI-Stream 首次响应: {api_ms:.1f}ms | provider={provider} | model={model}")
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _dashscope_chat(model: Optional[str], messages: List[Dict], cfg: dict = None):
    import dashscope
    from dashscope import Generation
    cfg = cfg or {}
    dashscope.api_key = cfg.get("api_key") or settings.DASHSCOPE_API_KEY
    model = model or cfg.get("model_name") or settings.DASHSCOPE_MODEL
    t_api = time.perf_counter()
    response = Generation.call(model=model, messages=messages, result_format="message")
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-DashScope API调用: {api_ms:.1f}ms | model={model} | msg_count={len(messages)}")
    content = response.output.choices[0].message.content
    return content, {}


async def _dashscope_chat_stream(model: Optional[str], messages: List[Dict], cfg: dict = None) -> AsyncGenerator[str, None]:
    import dashscope
    from dashscope import Generation
    cfg = cfg or {}
    dashscope.api_key = cfg.get("api_key") or settings.DASHSCOPE_API_KEY
    model = model or cfg.get("model_name") or settings.DASHSCOPE_MODEL
    t_api = time.perf_counter()
    responses = Generation.call(model=model, messages=messages, result_format="message", stream=True, incremental_output=True)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-DashScope-Stream 首次响应: {api_ms:.1f}ms | model={model}")
    for response in responses:
        delta = response.output.choices[0].message.content
        if delta:
            yield delta


async def _qianfan_chat(model: Optional[str], messages: List[Dict], cfg: dict = None):
    import qianfan
    cfg = cfg or {}
    chat_comp = qianfan.ChatCompletion(
        ak=cfg.get("api_key") or settings.QIANFAN_ACCESS_KEY,
        sk=cfg.get("api_secret") or settings.QIANFAN_SECRET_KEY,
    )
    model = model or cfg.get("model_name") or settings.QIANFAN_MODEL
    t_api = time.perf_counter()
    resp = await chat_comp.ado(model=model, messages=messages)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-Qianfan API调用: {api_ms:.1f}ms | model={model} | msg_count={len(messages)}")
    return resp.body["result"], {}


async def _ollama_chat(model: Optional[str], messages: List[Dict], stream: bool, cfg: dict = None):
    import ollama
    cfg = cfg or {}
    model = model or cfg.get("model_name") or settings.OLLAMA_MODEL
    base_url = cfg.get("base_url") or settings.OLLAMA_BASE_URL
    client = ollama.AsyncClient(host=base_url)
    t_api = time.perf_counter()
    response = await client.chat(model=model, messages=messages)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-Ollama API调用: {api_ms:.1f}ms | model={model} | base_url={base_url} | msg_count={len(messages)}")
    return response.message.content, {}


async def _ollama_chat_stream(model: Optional[str], messages: List[Dict], cfg: dict = None) -> AsyncGenerator[str, None]:
    import ollama
    cfg = cfg or {}
    model = model or cfg.get("model_name") or settings.OLLAMA_MODEL
    base_url = cfg.get("base_url") or settings.OLLAMA_BASE_URL
    client = ollama.AsyncClient(host=base_url)
    t_api = time.perf_counter()
    async for chunk in await client.chat(model=model, messages=messages, stream=True):
        api_ms = (time.perf_counter() - t_api) * 1000
        if api_ms < 1000 and chunk.message.content:
            logger.debug(f"LLM-Ollama-Stream 首次响应: {api_ms:.1f}ms | model={model} | base_url={base_url}")
            t_api = None
        delta = chunk.message.content
        if delta:
            yield delta


async def _lmstudio_chat(model: Optional[str], messages: List[Dict], stream: bool, cfg: dict = None):
    from openai import AsyncOpenAI
    cfg = cfg or {}
    base_url = cfg.get("base_url") or settings.LMSTUDIO_BASE_URL
    model = model or cfg.get("model_name") or settings.LMSTUDIO_MODEL
    client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
    t_api = time.perf_counter()
    response = await client.chat.completions.create(model=model, messages=messages, stream=False)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-LMStudio API调用: {api_ms:.1f}ms | model={model} | base_url={base_url} | msg_count={len(messages)}")
    return response.choices[0].message.content, {}


async def _lmstudio_chat_stream(model: Optional[str], messages: List[Dict], cfg: dict = None) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    cfg = cfg or {}
    base_url = cfg.get("base_url") or settings.LMSTUDIO_BASE_URL
    model = model or cfg.get("model_name") or settings.LMSTUDIO_MODEL
    client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
    t_api = time.perf_counter()
    stream = await client.chat.completions.create(model=model, messages=messages, stream=True)
    api_ms = (time.perf_counter() - t_api) * 1000
    logger.debug(f"LLM-LMStudio-Stream 首次响应: {api_ms:.1f}ms | model={model} | base_url={base_url}")
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
