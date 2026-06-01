import json
import redis.asyncio as aioredis
from app.core.config import settings
from app.core.logger import logger

_redis_pool = None


async def get_redis() -> aioredis.Redis | None:
    global _redis_pool
    if not settings.REDIS_ENABLED:
        return None
    if _redis_pool is None:
        try:
            _redis_pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=10,
                decode_responses=True,
            )
            r = aioredis.Redis(connection_pool=_redis_pool)
            await r.ping()
            logger.info(f"Redis 已连接: {settings.REDIS_URL}")
        except Exception as e:
            logger.warning(f"Redis 连接失败，缓存功能将跳过: {e}")
            return None
    return aioredis.Redis(connection_pool=_redis_pool)


async def close_redis():
    global _redis_pool
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None
        logger.info("Redis 连接已关闭")


async def cache_get(key: str) -> str | None:
    r = await get_redis()
    if r is None:
        return None
    try:
        return await r.get(key)
    except Exception:
        return None


async def cache_set(key: str, value: str, ttl: int = 300) -> bool:
    r = await get_redis()
    if r is None:
        return False
    try:
        await r.set(key, value, ex=ttl)
        return True
    except Exception:
        return False


async def cache_delete(key: str):
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(key)
    except Exception:
        pass


async def cache_delete_pattern(pattern: str):
    r = await get_redis()
    if r is None:
        return
    try:
        keys = []
        async for key in r.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            await r.delete(*keys)
    except Exception:
        pass


async def cache_get_json(key: str) -> dict | list | None:
    raw = await cache_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def cache_set_json(key: str, data: dict | list, ttl: int = 300) -> bool:
    return await cache_set(key, json.dumps(data, ensure_ascii=False), ttl)
