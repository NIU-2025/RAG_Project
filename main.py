import os
import time
from dotenv import load_dotenv
load_dotenv()

# 设置 HuggingFace 模型缓存目录（需在导入 sentence-transformers 之前生效）
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

hf_endpoint = os.getenv("HF_ENDPOINT")
if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint

os.makedirs("logs", exist_ok=True)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logger import logger
from app.db.session import init_db
from app.api.routes import auth, kb, docs, chat, search, models, voice, stats, role, user


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG 系统启动...")
    await init_db()
    logger.info("数据库初始化完成")

    # 预加载本地 Embedding 模型，避免首次请求时触发下载/加载
    if settings.EMBEDDING_PROVIDER == "local":
        import asyncio
        from app.services.embedding import EmbeddingService
        loop = asyncio.get_event_loop()
        logger.info(f"预加载本地 Embedding 模型: {settings.EMBEDDING_MODEL}，请稍候...")
        #异步加载模型，避免阻塞主进程。优先从本地缓存获取，若不存在则从 HuggingFace 下载。
        await loop.run_in_executor(None, EmbeddingService.preload_model)
        logger.info("Embedding 模型预加载完成")

    # 预加载 Reranker 模型
    if settings.RERANK_ENABLED:
        import asyncio
        from app.services.reranker import RerankerService
        loop = asyncio.get_event_loop()
        logger.info(f"预加载 Reranker 模型: {settings.RERANK_MODEL}，请稍候...")
        try:
            await loop.run_in_executor(None, RerankerService.preload)
            logger.info("Reranker 模型预加载完成")
        except Exception as e:
            logger.warning(f"Reranker 模型加载失败，重排序功能已禁用: {e}")
            settings.RERANK_ENABLED = False

    yield
    from app.core.redis_client import close_redis
    await close_redis()
    logger.info("RAG 系统关闭")


app = FastAPI(
    title="RAG 检索增强生成系统",
    description="基于 FastAPI + ChromaDB + MySQL 的 RAG 知识库问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（仅允许配置的前端地址）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 速率限制
if settings.RATE_LIMIT_ENABLED:
    try:
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        from slowapi.middleware import SlowAPIMiddleware
        from slowapi.errors import RateLimitExceeded

        _env_tmp = ".env._slowapi_tmp"
        _env_restore = False
        if os.path.exists(".env"):
            os.rename(".env", _env_tmp)
            _env_restore = True

        try:
            _limiter = Limiter(
                key_func=get_remote_address,
                default_limits=[settings.RATE_LIMIT_DEFAULT],
            )
        finally:
            if _env_restore:
                os.rename(_env_tmp, ".env")

        app.state.limiter = _limiter
        app.add_middleware(SlowAPIMiddleware)

        @app.exception_handler(RateLimitExceeded)
        async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
            )

        logger.info(f"速率限制已启用: {settings.RATE_LIMIT_DEFAULT}")
    except ImportError:
        logger.warning("slowapi 未安装，跳过速率限制。安装: pip install slowapi")


# 请求日志中间件
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed_ms:.1f}ms)"
    )
    return response

# 注册路由
app.include_router(auth.router, prefix="/api/v1")
app.include_router(kb.router, prefix="/api/v1")
app.include_router(docs.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")
app.include_router(stats.router, prefix="/api/v1")
app.include_router(role.router, prefix="/api/v1")
app.include_router(user.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "RAG System API", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.APP_ENV}


# ---------- 前端静态文件挂载（SPA）----------
dist_path = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(dist_path):
    app.mount("/static", StaticFiles(directory=dist_path, html=True), name="static")

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        index_path = os.path.join(dist_path, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"error": "Not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
