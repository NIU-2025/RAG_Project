"""
RAGAS 评估脚本
基于 30 条标注测试集，对 RAG 系统进行 4 维度量化评估。

用法:
    cd d:/AI_code/RAGProject
    python tests/evaluation/evaluate_rag.py

输出:
    - 终端打印评估报告
    - tests/evaluation/eval_report.md 保存 Markdown 报告
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

hf_endpoint = os.getenv("HF_ENDPOINT")
if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("RERANK_ENABLED", "false")

from app.core.config import settings
from app.core.logger import logger
from app.db.session import AsyncSessionLocal
from app.services.retrieval import RetrievalService
from app.services.llm import LLMService
from app.services.embedding import EmbeddingService

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH = Path(__file__).parent / "eval_report.md"
REPORT_JSON_PATH = Path(__file__).parent / "eval_report.json"
BASELINE_PATH = Path(__file__).parent / "baseline.json"
KB_ID = 1

# ──────────────────────────────────────────────────────
# RAG 流水线：检索 → 生成
# ──────────────────────────────────────────────────────


async def run_rag(query: str, top_k: int = 5, query_embedding: list = None) -> dict:
    """单次 RAG 问答：检索 + 生成，支持传入预计算 embedding 加速评估"""
    t_start = time.perf_counter()

    # 检索（支持外部传入预计算 embedding）
    t_retrieval = time.perf_counter()
    async with AsyncSessionLocal() as db:
        if query_embedding is not None:
            search_results = await RetrievalService.search_with_embedding(
                kb_id=KB_ID,
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                db=db,
            )
        else:
            search_results = await RetrievalService.search(
                kb_id=KB_ID,
                query=query,
                top_k=top_k,
                db=db,
            )
    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

    contexts = [r.content for r in search_results] if search_results else []
    context_str = ""
    if search_results:
        parts = []
        for i, r in enumerate(search_results, 1):
            parts.append(f"[{i}] 来源：{r.filename}（第{r.page_num or '-'}页）\n{r.content}")
        context_str = "\n\n".join(parts)

    # 生成
    t_llm = time.perf_counter()
    try:
        answer, _ = await LLMService.chat(
            provider=settings.DEFAULT_LLM_PROVIDER,
            model=None,
            messages=[],
            user_message=query,
            context=context_str,
        )
    except Exception as e:
        answer = f"[LLM 调用失败: {e}]"
    llm_ms = (time.perf_counter() - t_llm) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    return {
        "contexts": contexts,
        "answer": answer or "",
        "retrieval_ms": retrieval_ms,
        "llm_ms": llm_ms,
        "total_ms": total_ms,
    }


# ──────────────────────────────────────────────────────
# RAGAS 评估核心
# ──────────────────────────────────────────────────────


def _build_ragas_dataset(records: list[dict]) -> "Dataset":
    from datasets import Dataset

    return Dataset.from_dict({
        "question": [r["question"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })


def run_ragas_evaluation(records: list[dict]) -> dict:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    )
    from openai import OpenAI
    from ragas.llms import llm_factory

    client = OpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    evaluator_llm = llm_factory(
        getattr(settings, "DASHSCOPE_MODEL", "qwen-plus"),
        client=client,
    )

    ds = _build_ragas_dataset(records)
    result = evaluate(
        ds,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=evaluator_llm,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c in (
        "context_precision", "context_recall", "faithfulness", "answer_relevancy",
    )]
    scores = {}
    for col in metric_cols:
        col_mean = df[col].dropna().mean()
        scores[col] = round(float(col_mean), 4)

    detail_rows = []
    for i, row in df.iterrows():
        detail_rows.append({
            "id": records[i]["id"],
            "category": records[i]["category"],
            "question": records[i]["question"][:40],
            "context_precision": round(float(row.get("context_precision", 0)), 4),
            "context_recall": round(float(row.get("context_recall", 0)), 4),
            "faithfulness": round(float(row.get("faithfulness", 0)), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0)), 4),
        })

    return {
        "summary": scores,
        "detail": detail_rows,
    }


# ──────────────────────────────────────────────────────
# 报告输出
# ──────────────────────────────────────────────────────


def print_report(summary: dict, detail: list, records: list, rerank_enabled: bool):
    """终端打印评估报告"""
    total = len(records)
    avg_retrieval = sum(r.get("retrieval_ms", 0) for r in records) / max(total, 1)
    avg_llm = sum(r.get("llm_ms", 0) for r in records) / max(total, 1)
    avg_total = sum(r.get("total_ms", 0) for r in records) / max(total, 1)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  RAG 系统评估报告 — 员工考勤管理制度")
    print(f"  测试集条数: {total} | 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Reranker: {settings.RERANK_MODEL} {'(启用)' if rerank_enabled else '(关闭)'}")
    print(f"  评测LLM: dashscope/{getattr(settings, 'DASHSCOPE_MODEL', 'qwen-plus')} | 生成LLM: {settings.DEFAULT_LLM_PROVIDER}")
    print(f"{sep}")
    for k, v in summary.items():
        print(f"  {k:<24s}: {v:.4f}")
    print(f"  {'─' * 60}")
    print(f"  平均检索耗时             : {avg_retrieval:.1f}ms")
    print(f"  平均 LLM 耗时            : {avg_llm:.1f}ms")
    print(f"  平均端到端耗时           : {avg_total:.1f}ms")
    print(f"{sep}")

    # 按分类汇总
    cat_scores = {}
    for row in detail:
        cat = row["category"]
        cat_scores.setdefault(cat, {"count": 0, "precision": 0, "recall": 0, "faithfulness": 0, "relevancy": 0})
        d = cat_scores[cat]
        d["count"] += 1
        d["precision"] += row["context_precision"]
        d["recall"] += row["context_recall"]
        d["faithfulness"] += row["faithfulness"]
        d["relevancy"] += row["answer_relevancy"]

    print(f"\n  [Category] 按类别分析:")
    print(f"  {'类别':<16s} {'条数':>4s}  {'Precision':>10s}  {'Recall':>8s}  {'Faithful':>9s}  {'Relevancy':>9s}")
    print(f"  {'─' * 60}")
    for cat, d in cat_scores.items():
        n = d["count"]
        print(f"  {cat:<16s} {n:>4d}  {d['precision']/n:>10.4f}  {d['recall']/n:>8.4f}  {d['faithfulness']/n:>9.4f}  {d['relevancy']/n:>9.4f}")


def save_markdown(summary: dict, detail: list, records: list, rerank_enabled: bool):
    """保存 Markdown 评估报告"""
    total = len(records)
    avg_retrieval = sum(r.get("retrieval_ms", 0) for r in records) / max(total, 1)
    avg_llm = sum(r.get("llm_ms", 0) for r in records) / max(total, 1)
    avg_total = sum(r.get("total_ms", 0) for r in records) / max(total, 1)

    lines = []
    lines.append("# RAG 系统评估报告\n")
    lines.append(f"> 测试集: {total} 条 | 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Reranker: `{settings.RERANK_MODEL}` {'[OK] 启用' if rerank_enabled else '[ERROR] 关闭'}")
    lines.append(f"> LLM: `{settings.DEFAULT_LLM_PROVIDER}` | 评测模型: `dashscope/{getattr(settings, 'DASHSCOPE_MODEL', 'qwen-plus')}`\n")

    lines.append("## [RAGAS] 综合指标\n")
    lines.append("| 指标 | 得分 |")
    lines.append("|------|------|")
    for k, v in summary.items():
        lines.append(f"| {k} | {v:.4f} |")
    lines.append(f"| 平均检索耗时 | {avg_retrieval:.1f}ms |")
    lines.append(f"| 平均 LLM 耗时 | {avg_llm:.1f}ms |")
    lines.append(f"| 平均端到端耗时 | {avg_total:.1f}ms |\n")

    # 分类汇总
    cat_scores = {}
    for row in detail:
        cat = row["category"]
        cat_scores.setdefault(cat, {"count": 0, "precision": 0, "recall": 0, "faithfulness": 0, "relevancy": 0})
        d = cat_scores[cat]
        d["count"] += 1
        d["precision"] += row["context_precision"]
        d["recall"] += row["context_recall"]
        d["faithfulness"] += row["faithfulness"]
        d["relevancy"] += row["answer_relevancy"]

    lines.append("## [Category] 按类别分析\n")
    lines.append("| 类别 | 条数 | context_precision | context_recall | faithfulness | answer_relevancy |")
    lines.append("|------|------|------------------|---------------|--------------|------------------|")
    for cat, d in cat_scores.items():
        n = d["count"]
        lines.append(f"| {cat} | {n} | {d['precision']/n:.4f} | {d['recall']/n:.4f} | {d['faithfulness']/n:.4f} | {d['relevancy']/n:.4f} |")

    lines.append("\n## [Detail] 逐条详情\n")
    lines.append("| # | 类别 | 问题 | c_precision | c_recall | faithful | relevancy |")
    lines.append("|---|------|------|------------|---------|----------|----------|")
    for row in detail:
        lines.append(
            f"| {row['id']} | {row['category']} | {row['question']} "
            f"| {row['context_precision']:.4f} | {row['context_recall']:.4f} "
            f"| {row['faithfulness']:.4f} | {row['answer_relevancy']:.4f} |"
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [Report] 报告已保存: {REPORT_PATH}")


def save_json_report(summary: dict, detail: list, records: list, rerank_enabled: bool):
    """保存 JSON 格式评估结果，用于自动对比"""
    total = len(records)
    avg_retrieval = sum(r.get("retrieval_ms", 0) for r in records) / max(total, 1)
    avg_llm = sum(r.get("llm_ms", 0) for r in records) / max(total, 1)
    avg_total = sum(r.get("total_ms", 0) for r in records) / max(total, 1)

    data = {
        "version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": EVAL_TOP_K,
            "rerank_enabled": rerank_enabled,
            "rerank_model": settings.RERANK_MODEL if rerank_enabled else None,
            "eval_llm": f"dashscope/{getattr(settings, 'DASHSCOPE_MODEL', 'qwen-plus')}",
            "gen_llm": settings.DEFAULT_LLM_PROVIDER,
        },
        "summary": summary,
        "detail": detail,
        "perf": {
            "avg_retrieval_ms": round(avg_retrieval, 1),
            "avg_llm_ms": round(avg_llm, 1),
            "avg_total_ms": round(avg_total, 1),
        },
    }
    REPORT_JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [Report] JSON 报告已保存: {REPORT_JSON_PATH}")


def save_baseline():
    """将当前 JSON 报告另存为基线"""
    import shutil
    if REPORT_JSON_PATH.exists():
        shutil.copy(REPORT_JSON_PATH, BASELINE_PATH)
        print(f"  [Baseline] 基线已保存: {BASELINE_PATH}")
    else:
        print(f"  [ERROR] 未找到 JSON 报告，请先运行评估")


# ──────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────


MAX_CONCURRENT = 1
EVAL_TOP_K = 8
EMBED_BATCH_SIZE = 1


async def _warm_bm25_cache(kb_id: int):
    """预热 BM25 分词语料缓存，避免评估时逐条重建"""
    try:
        async with AsyncSessionLocal() as db:
            await RetrievalService.search(
                kb_id=kb_id,
                query="预热",
                top_k=1,
                db=db,
            )
        logger.info("BM25 缓存预热完成")
    except Exception as e:
        logger.warning(f"BM25 缓存预热跳过: {e}")


async def _run_one(item: dict, idx: int, total: int, sem: asyncio.Semaphore,
                   query_embeddings: dict = None) -> dict:
    async with sem:
        q = item["question"]
        q_emb = query_embeddings.get(q) if query_embeddings else None
        result = await run_rag(q, top_k=EVAL_TOP_K, query_embedding=q_emb)
        ctx_count = len(result["contexts"])
        ans_len = len(result["answer"])
        print(f"  [{idx:2d}/{total}] [OK] {result['total_ms']:.0f}ms | "
              f"ctx={ctx_count} | ans={ans_len}字 | {q[:30]}")
        return {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "contexts": result["contexts"],
            "answer": result["answer"],
            "retrieval_ms": result["retrieval_ms"],
            "llm_ms": result["llm_ms"],
            "total_ms": result["total_ms"],
        }


async def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"\n[OK] 加载测试集: {len(dataset)} 条")

    # ═══ 加速步骤 1：分批预计算所有 query embedding ═══
    print(f"\n[...] 分批预计算 Query Embedding ({len(dataset)} 条, 每批 {EMBED_BATCH_SIZE} 条)...")
    all_queries = [item["question"] for item in dataset]
    t_embed = time.perf_counter()
    all_embeddings = []
    for i in range(0, len(all_queries), EMBED_BATCH_SIZE):
        batch = all_queries[i:i + EMBED_BATCH_SIZE]
        batch_embs = EmbeddingService.embed_texts(batch)
        all_embeddings.extend(batch_embs)
        print(f"  Embedding 批次 {i//EMBED_BATCH_SIZE + 1}/{(len(all_queries)-1)//EMBED_BATCH_SIZE + 1}: {len(batch)} 条")
    query_embeddings = {q: emb for q, emb in zip(all_queries, all_embeddings)}
    embed_total_ms = (time.perf_counter() - t_embed) * 1000
    print(f"[OK] Query Embedding 完成: {embed_total_ms:.0f}ms ({embed_total_ms/len(all_queries):.0f}ms/条)")

    # ═══ 加速步骤 2：预热 BM25 缓存 ═══
    print(f"\n[...] 预热 BM25 分词语料缓存...")
    t_bm25 = time.perf_counter()
    await _warm_bm25_cache(KB_ID)
    bm25_ms = (time.perf_counter() - t_bm25) * 1000
    print(f"[OK] BM25 缓存预热完成: {bm25_ms:.0f}ms")

    # ═══ 并行执行 RAG 流水线 ═══
    print(f"\n[...] 开始 RAG 流水线评估 ({len(dataset)} 条, 并发={MAX_CONCURRENT}, top_k={EVAL_TOP_K})...\n")

    total = len(dataset)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [_run_one(item, i + 1, total, sem, query_embeddings) for i, item in enumerate(dataset)]
    records = await asyncio.gather(*tasks)

    # 有检索结果的条目进入 RAGAS 评估
    ragas_records = [r for r in records if r["contexts"]]

    print(f"\n[RAGAS] 开始 RAGAS 评估（评测 LLM: dashscope/{getattr(settings, 'DASHSCOPE_MODEL', 'qwen-plus')})...")
    try:
        eval_result = run_ragas_evaluation(ragas_records)
        # 检查并标记 NaN 指标
        nan_metrics = [k for k, v in eval_result["summary"].items() if v != v]
        if nan_metrics:
            print(f"\n[WARN]  警告：以下指标包含 NaN（评测 LLM 部分条目解析失败）: {', '.join(nan_metrics)}")
            print(f"   建议：若 NaN 比例高，可尝试换用 qwen-max 或 Ollama 本地模型")
    except Exception as e:
        print(f"\n[ERROR] RAGAS 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return

    rerank_enabled = getattr(settings, "RERANK_ENABLED", False)
    print_report(eval_result["summary"], eval_result["detail"], records, rerank_enabled)
    save_markdown(eval_result["summary"], eval_result["detail"], records, rerank_enabled)
    save_json_report(eval_result["summary"], eval_result["detail"], records, rerank_enabled)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--save-baseline":
        asyncio.run(main())
        save_baseline()
    else:
        asyncio.run(main())
