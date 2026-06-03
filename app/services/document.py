import uuid
import os
from concurrent.futures import ProcessPoolExecutor
import asyncio
from sqlalchemy import select
from app.models.db import Document, DocumentChunk, KnowledgeBase, DocumentStatus
from app.db.session import AsyncSessionLocal
from app.db.vector_store import VectorStore
from app.services.embedding import EmbeddingService
from app.core.config import settings
from app.core.logger import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

_process_pool = None


def _get_process_pool():
    """获取进程池（CPU 核数 - 1 个 worker，最少 2 个）"""
    global _process_pool
    if _process_pool is None:
        workers = max(2, (os.cpu_count() or 4) - 1)
        _process_pool = ProcessPoolExecutor(max_workers=workers)
        logger.info(f"文档处理进程池已创建，workers={workers}")
    return _process_pool


# =============================================================================
# 进程池中执行的同步函数（所有 CPU/IO 密集型操作在这里）
# =============================================================================

def _sync_process_doc(doc_id: int):
    """
    同步函数，运行在进程池中，执行文档解析 → 分块 → 向量化 → ChromaDB 写入。
    完成后返回 (True, chunk_count) 或 (False, error_msg)。
    """
    from sqlalchemy.orm import Session
    from app.db.session import sync_engine
    from app.models.db import Document, DocumentChunk, KnowledgeBase, DocumentStatus
    from app.parsers import get_parser
    from app.services.embedding import EmbeddingService
    from app.db.vector_store import VectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import uuid as _uuid

    session = Session(sync_engine)
    try:
        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()

        if not doc:
            return False, "文档不存在"

        logger.info(f"[进程池] 开始处理文档 [{doc_id}]: {doc.filename}")

        # 1. 解析文档
        parser = get_parser(doc.file_type)
        parsed = parser.parse(doc.file_path)

        if not parsed.pages:
            return False, "文档解析结果为空"

        # 2. 分块（block-aware：表格/OCR 作为原子 chunk，纯文本按递归切分）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

        all_chunks = []  # (content, page_num, source_type)
        for page in parsed.pages:
            if page.blocks:
                # 新解析器：按 ContentBlock 类型差异化处理
                for block in page.blocks:
                    if not block.content.strip():
                        continue
                    if block.type == "table":
                        # 表格作为原子 chunk，不跨表切分
                        all_chunks.append((block.content.strip(), page.page_num, "table"))
                    elif block.type == "image_text":
                        # OCR 文字作为原子 chunk
                        all_chunks.append((block.content.strip(), page.page_num, "image_text"))
                    else:
                        # 纯文本：正常递归切分
                        splits = splitter.split_text(block.content)
                        for split in splits:
                            if split.strip():
                                all_chunks.append((split.strip(), page.page_num, "text"))
            else:
                # 旧解析器（无 blocks，向后兼容）
                if not page.content.strip():
                    continue
                splits = splitter.split_text(page.content)
                for split in splits:
                    if split.strip():
                        all_chunks.append((split.strip(), page.page_num, "text"))

        if not all_chunks:
            return False, "文档分块结果为空"

        logger.info(f"[进程池] 文档 [{doc_id}] 分块完成，共 {len(all_chunks)} 块")

        # 3. 向量化
        texts = [c[0] for c in all_chunks]
        embeddings = EmbeddingService.embed_texts(texts)

        # 4. 构建 ChromaDB 数据
        chroma_ids = []
        chroma_metadatas = []
        db_chunks = []

        for i, ((content, page_num, source_type), embedding) in enumerate(zip(all_chunks, embeddings)):
            chroma_id = f"doc{doc_id}_chunk{i}_{_uuid.uuid4().hex[:8]}"
            chroma_ids.append(chroma_id)
            chroma_metadatas.append({
                "doc_id": doc_id,
                "kb_id": doc.kb_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "page_num": page_num or 0,
                "chunk_index": i,
                "source_type": source_type,
                "tags": doc.tags or "",
            })
            db_chunks.append(DocumentChunk(
                doc_id=doc_id,
                kb_id=doc.kb_id,
                chroma_id=chroma_id,
                content=content,
                chunk_index=i,
                page_num=page_num,
                token_count=len(content),
            ))

        # 5. 写入 ChromaDB
        VectorStore.add_documents(
            kb_id=doc.kb_id,
            ids=chroma_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=chroma_metadatas,
        )

        # 6. 写入 MySQL chunks
        for chunk in db_chunks:
            session.add(chunk)

        # 更新文档状态
        doc.status = DocumentStatus.completed
        doc.chunk_count = len(all_chunks)

        # 更新知识库文档数
        kb = session.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
        ).scalar_one_or_none()
        if kb:
            kb.doc_count = (kb.doc_count or 0) + 1

        session.commit()
        logger.info(f"[进程池] 文档 [{doc_id}] 处理完成，{len(all_chunks)} 个向量已入库")
        return True, len(all_chunks)

    except Exception as e:
        session.rollback()
        err_msg = str(e)
        if "未识别到文字内容" in err_msg or "图片文件无效" in err_msg or "图片中未识别到文字" in err_msg:
            err_msg = "图片中未识别到文字内容，无法入库，请上传包含文字的图片"
        logger.error(f"[进程池] 文档 [{doc_id}] 处理失败: {err_msg}")

        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()
        if doc:
            doc.status = DocumentStatus.failed
            doc.error_msg = err_msg[:500]
            session.commit()

        return False, err_msg

    finally:
        session.close()


def _sync_process_audio_doc(doc_id: int, asr_text: str):
    """处理 ASR 识别文本（分块+向量化），运行在进程池中"""
    from sqlalchemy.orm import Session
    from app.db.session import sync_engine
    from app.models.db import Document, KnowledgeBase, DocumentStatus, DocumentChunk
    from app.services.embedding import EmbeddingService
    from app.db.vector_store import VectorStore
    from app.parsers.voice import VoiceParser
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import uuid as _uuid

    session = Session(sync_engine)
    try:
        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()
        if not doc:
            return

        parsed = VoiceParser(asr_text=asr_text).parse(doc.file_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

        all_chunks = []
        for page in parsed.pages:
            if not page.content.strip():
                continue
            splits = splitter.split_text(page.content)
            for split in splits:
                if split.strip():
                    all_chunks.append((split.strip(), page.page_num or 0, "text"))

        if not all_chunks:
            raise ValueError("文档分块结果为空")

        texts = [c[0] for c in all_chunks]
        embeddings = EmbeddingService.embed_texts(texts)

        chroma_ids = []
        chroma_metadatas = []
        for i, ((content, page_num, source_type), embedding) in enumerate(zip(all_chunks, embeddings)):
            chroma_id = f"doc{doc_id}_chunk{i}_{_uuid.uuid4().hex[:8]}"
            chroma_ids.append(chroma_id)
            chroma_metadatas.append({
                "doc_id": doc_id,
                "kb_id": doc.kb_id,
                "filename": doc.filename,
                "file_type": doc.file_type,
                "page_num": page_num,
                "chunk_index": i,
                "source_type": source_type,
                "tags": doc.tags or "",
            })

        VectorStore.add_documents(
            kb_id=doc.kb_id,
            ids=chroma_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=chroma_metadatas,
        )

        for i, (content, page_num, _) in enumerate(all_chunks):
            session.add(DocumentChunk(
                doc_id=doc_id,
                kb_id=doc.kb_id,
                chroma_id=chroma_ids[i],
                content=content,
                chunk_index=i,
                page_num=page_num,
                token_count=len(content),
            ))

        doc.status = DocumentStatus.completed
        doc.chunk_count = len(all_chunks)
        kb = session.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == doc.kb_id)
        ).scalar_one_or_none()
        if kb:
            kb.doc_count = (kb.doc_count or 0) + 1

        session.commit()
        logger.info(f"[进程池] 语音文档 [{doc_id}] 处理完成，{len(all_chunks)} 个向量已入库")

    except Exception as e:
        session.rollback()
        logger.error(f"[进程池] 语音文档 [{doc_id}] 处理失败: {e}")
        doc = session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()
        if doc:
            doc.status = DocumentStatus.failed
            doc.error_msg = str(e)[:500]
            session.commit()
    finally:
        session.close()


# =============================================================================
# 语音识别（线程池执行，支撑 async 语音服务）
# =============================================================================

def _run_async_in_thread(coro):
    """在当前线程中运行协程（用于 ThreadPoolExecutor 中调用 async 函数）"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已有事件循环，直接运行
            future = asyncio.ensure_future(coro)
            return loop.run_until_complete(future)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # 没有事件循环，创建新的
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _sync_voice_asr(doc_id: int, provider: str, api_key: str, api_secret: str, extra: dict) -> str:
    """在线程池中执行同步 ASR（内部通过新事件循环调用 async voice service）"""
    from app.services.voice_asr import VoiceASRService
    from sqlalchemy.orm import Session
    from app.db.session import sync_engine
    from app.models.db import Document
    from sqlalchemy import select

    # 从 DB 读取文件路径
    session = Session(sync_engine)
    try:
        doc = session.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
        file_path = doc.file_path if doc else ""
    finally:
        session.close()

    if not file_path:
        raise RuntimeError(f"文档 {doc_id} 文件路径不存在")

    # 构造协程，在新线程的事件循环中运行
    async def _do():
        return await VoiceASRService.recognize(
            file_path=file_path,
            provider=provider,
            api_key=api_key,
            api_secret=api_secret,
            extra_params=extra,
        )

    return _run_async_in_thread(_do())


# =============================================================================
# ASGI 事件循环中调用的 async 入口
# =============================================================================

async def _do_voice_asr(doc_id: int, provider: str, api_key: str, api_secret: str, extra: dict) -> str:
    """在事件循环中调用，在线程池执行同步 ASR"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        ThreadPoolExecutor(max_workers=1),
        _sync_voice_asr,
        doc_id, provider, api_key, api_secret, extra
    )


# =============================================================================
# 文档处理服务（async 入口，供路由层调用）
# =============================================================================

class DocumentService:

    @staticmethod
    async def _get_voice_config(doc_id: int) -> dict:
        """查询已启用的语音配置"""
        from app.models.db import VoiceConfig
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(VoiceConfig)
                .where(VoiceConfig.is_enabled == True)
                .order_by(VoiceConfig.is_default.desc())
                .limit(1)
            )
            cfg = result.scalar_one_or_none()
        return cfg

    @staticmethod
    async def process_document(doc_id: int):
        """
        文档处理入口（async，供 asyncio.create_task 调用）。
        流程：查询文档信息 → 路由到进程池执行 CPU 密集型任务。
        事件循环始终保持空闲，不阻塞其他请求。
        """
        loop = asyncio.get_running_loop()
        pool = _get_process_pool()

        # 查询文档基本信息
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if not doc:
                return
            kb_id = doc.kb_id
            file_type = doc.file_type
            file_path = doc.file_path

        AUDIO_TYPES = {"mp3", "wav", "m4a", "aac", "ogg", "wma", "flac", "pcm"}

        if file_type.lower() in AUDIO_TYPES:
            # 语音文件：先 ASR，再进进程池处理
            cfg = await DocumentService._get_voice_config(doc_id)
            if not cfg:
                await _mark_failed_async(doc_id, "未找到已启用的语音识别配置，请前往「语音配置」页面添加并启用")
                return

            import json as _json
            extra = _json.loads(cfg.extra_params) if cfg.extra_params else {}

            # 验证配置完整性
            missing = []
            if not cfg.api_key: missing.append("API Key")
            if not cfg.api_secret: missing.append("Secret Key")
            if cfg.provider == "baidu" and not extra.get("app_id"): missing.append("App ID")
            if cfg.provider == "aliyun" and not extra.get("app_key"): missing.append("AppKey")
            if missing:
                await _mark_failed_async(doc_id, f"{cfg.provider} ASR 配置不完整，缺少: {', '.join(missing)}")
                return

            try:
                logger.info(f"[voice] 开始 ASR 识别文档 [{doc_id}]，provider={cfg.provider}")
                asr_text = await _do_voice_asr(doc_id, cfg.provider, cfg.api_key, cfg.api_secret, extra)
                logger.info(f"[voice] ASR 识别完成，文档 [{doc_id}]，结果长度={len(asr_text)}")
            except Exception as e:
                await _mark_failed_async(doc_id, f"语音识别失败: {e}")
                return

            # ASR 文本进进程池处理（分块+向量化）
            try:
                await loop.run_in_executor(pool, _sync_process_audio_doc, doc_id, asr_text)
                VectorStore._client = None
                VectorStore.get_client()
                await _invalidate_bm25_cache(kb_id)
            except Exception as e:
                await _mark_failed_async(doc_id, str(e))
        else:
            try:
                await loop.run_in_executor(pool, _sync_process_doc, doc_id)
                VectorStore._client = None
                VectorStore.get_client()
                await _invalidate_bm25_cache(kb_id)
            except Exception as e:
                await _mark_failed_async(doc_id, str(e))


async def _invalidate_bm25_cache(kb_id: int):
    """文档入库后失效 BM25 缓存"""
    try:
        from app.core.redis_client import cache_delete
        await cache_delete(f"kb:{kb_id}:bm25")
        logger.debug(f"BM25 缓存已失效: kb_id={kb_id}")
    except Exception:
        pass


async def _mark_failed_async(doc_id: int, err_msg: str):
    """异步方式将文档状态标记为失败"""
    if "未识别到文字内容" in err_msg or "图片文件无效" in err_msg or "图片中未识别到文字" in err_msg:
        err_msg = "图片中未识别到文字内容，无法入库，请上传包含文字的图片"
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = DocumentStatus.failed
                doc.error_msg = err_msg[:500]
                await db.commit()
    except Exception:
        pass
