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
import re
import sys
import time
from pathlib import Path
from typing import List

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

import numpy as np

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

EVAL_THRESHOLD = 0.38  # embedding 相似度阈值，用于判断语义相关/忠实

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
# 自定义评估指标（本地 embedding 版，零 LLM 依赖，永不 NaN）
# ──────────────────────────────────────────────────────


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """余弦相似度"""
    a_norm = np.array(a) / (np.linalg.norm(a) + 1e-12)
    b_norm = np.array(b) / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def _compute_answer_relevancy(question: str, answer: str) -> float:
    """基于 embedding 的答案相关性：问题和答案的语义相似度"""
    if not answer or not question:
        return 0.0
    q_emb = EmbeddingService.embed_query(question)
    a_emb = EmbeddingService.embed_query(answer)
    return _cosine_sim(q_emb, a_emb)


def _compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    基于 embedding 的忠实度。
    将答案拆分为句子，检查每句是否在上下文中找到语义支撑（> 阈值）。
    """
    if not answer:
        return 0.0
    if not contexts or all(not c for c in contexts):
        return 1.0  # 无上下文可违背 → 完全忠实

    sentences = [s.strip() for s in re.split(r'[。！？\n;；]', answer) if len(s.strip()) > 2]
    if not sentences:
        return 1.0

    valid_contexts = [c for c in contexts if c]
    if not valid_contexts:
        return 1.0

    c_embs = EmbeddingService.embed_texts(valid_contexts)
    s_embs = EmbeddingService.embed_texts(sentences)

    c_arr = np.array(c_embs)
    s_arr = np.array(s_embs)
    c_norm = c_arr / (np.linalg.norm(c_arr, axis=1, keepdims=True) + 1e-12)
    s_norm = s_arr / (np.linalg.norm(s_arr, axis=1, keepdims=True) + 1e-12)

    # sim: (n_sentences, n_contexts)
    sim_matrix = np.dot(s_norm, c_norm.T)
    max_sims = np.max(sim_matrix, axis=1)

    # 滑动窗口平滑（窗口=3）：每句得分 = 自身 + 相邻句的平均
    # 过渡句（如"具体规则如下"）前后都是内容句，窗口平滑后自然拉升
    window_size = 3
    n = len(max_sims)
    smoothed = np.copy(max_sims)
    for i in range(n):
        left = max(0, i - window_size // 2)
        right = min(n, i + window_size // 2 + 1)
        smoothed[i] = float(np.mean(max_sims[left:right]))

    supported = np.sum(smoothed >= EVAL_THRESHOLD)
    return float(supported / len(sentences))


def _compute_retrieval_metrics(records: List[dict]) -> dict:
    """
    计算检索质量指标：Recall@K, Precision@K
    用 ground_truth 拆句后与 contexts 做 embedding 相似度判断。
    """
    recalls, precisions = [], []
    for r in records:
        gt = r.get("ground_truth", "")
        contexts = r.get("contexts", [])
        if not gt or not contexts:
            recalls.append(0.0)
            precisions.append(0.0)
            continue

        gt_sents = [s.strip() for s in re.split(r'[。！？\n;；]', gt) if len(s.strip()) > 2]
        if not gt_sents:
            gt_sents = [gt]

        gt_embs = EmbeddingService.embed_texts(gt_sents)
        c_embs = EmbeddingService.embed_texts([c for c in contexts if c])
        if not gt_embs or not c_embs:
            recalls.append(0.0)
            precisions.append(0.0)
            continue

        gt_arr = np.array(gt_embs)
        c_arr = np.array(c_embs)
        gt_norm = gt_arr / (np.linalg.norm(gt_arr, axis=1, keepdims=True) + 1e-12)
        c_norm = c_arr / (np.linalg.norm(c_arr, axis=1, keepdims=True) + 1e-12)

        sim = np.dot(gt_norm, c_norm.T)  # (n_gt, n_contexts)
        covered = np.max(sim, axis=1) >= EVAL_THRESHOLD
        context_rel = np.max(sim, axis=0) >= EVAL_THRESHOLD

        recalls.append(float(np.mean(covered)))
        precisions.append(float(np.mean(context_rel)))

    return {
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "precision_at_k": round(float(np.mean(precisions)), 4),
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
    """
    全 embedding 评估策略（零 LLM 依赖）：
      - ContextPrecision / ContextRecall：用 query vs context 的 embedding 余弦相似度
      - Faithfulness / AnswerRelevancy：用自定义 embedding 版
      - 额外输出检索指标 Recall@K / Precision@K
    """
    # ── 1. ContextPrecision & ContextRecall（embedding 版）──
    ctx_precision = []
    ctx_recall = []
    for r in records:
        query = r["question"]
        contexts = r.get("contexts", [])
        if not contexts or all(not c for c in contexts):
            ctx_precision.append(0.0)
            ctx_recall.append(0.0)
            continue

        q_emb = EmbeddingService.embed_query(query)
        c_embs = EmbeddingService.embed_texts([c for c in contexts if c])
        if not c_embs:
            ctx_precision.append(0.0)
            ctx_recall.append(0.0)
            continue

        sims = [_cosine_sim(q_emb, ce) for ce in c_embs]
        # precision: 相关 context 的比例
        relevant = sum(1 for s in sims if s >= EVAL_THRESHOLD)
        ctx_precision.append(round(relevant / len(sims), 4))
        # recall: 如果至少有一个相关 context → 1.0
        ctx_recall.append(round(1.0 if relevant > 0 else 0.0, 4))

    # ── 2. Faithfulness & AnswerRelevancy（embedding 版）──
    custom_faithfulness = []
    custom_relevancy = []
    for r in records:
        custom_faithfulness.append(
            round(_compute_faithfulness(r["answer"], r.get("contexts", [])), 4)
        )
        custom_relevancy.append(
            round(_compute_answer_relevancy(r["question"], r["answer"]), 4)
        )

    # ── 3. 检索指标 ──
    retrieval_metrics = _compute_retrieval_metrics(records)

    # ── 4. 合并结果 ──
    summary = {
        "context_precision": round(float(np.mean(ctx_precision)), 4),
        "context_recall": round(float(np.mean(ctx_recall)), 4),
        "faithfulness": round(float(np.mean(custom_faithfulness)), 4),
        "answer_relevancy": round(float(np.mean(custom_relevancy)), 4),
        "retrieval/recall@k": retrieval_metrics["recall_at_k"],
        "retrieval/precision@k": retrieval_metrics["precision_at_k"],
    }

    detail_rows = []
    for i, r in enumerate(records):
        detail_rows.append({
            "id": r["id"],
            "category": r["category"],
            "question": r["question"][:40],
            "context_precision": ctx_precision[i] if i < len(ctx_precision) else 0.0,
            "context_recall": ctx_recall[i] if i < len(ctx_recall) else 0.0,
            "faithfulness": custom_faithfulness[i] if i < len(custom_faithfulness) else 0.0,
            "answer_relevancy": custom_relevancy[i] if i < len(custom_relevancy) else 0.0,
        })

    return {
        "summary": summary,
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
    print(f"  生成LLM: {settings.DEFAULT_LLM_PROVIDER}")
    print(f"{sep}")

    # 分类显示
    print(f"  ── [Embedding] 语义相似度指标──")
    for k in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if k in summary:
            print(f"  {k:<24s}: {summary[k]:.4f}")

    print(f"  ── [Retrieval] 检索质量 ──")
    for k in ["retrieval/recall@k", "retrieval/precision@k"]:
        if k in summary:
            print(f"  {k:<24s}: {summary[k]:.4f}")

    print(f"  {'─' * 60}")
    print(f"  平均检索耗时             : {avg_retrieval:.1f}ms")
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
    lines.append(f"> LLM: `{settings.DEFAULT_LLM_PROVIDER}` | 评估方法: `本地 Embedding 余弦相似度`\n")

    lines.append("## 综合指标\n")
    lines.append("### 语义相似度指标\n")
    lines.append("| 指标 | 得分 |")
    lines.append("|------|------|")
    for k in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if k in summary:
            lines.append(f"| {k} | {summary[k]:.4f} |")

    lines.append("\n### 检索质量\n")
    lines.append("| 指标 | 得分 |")
    lines.append("|------|------|")
    for k in ["retrieval/recall@k", "retrieval/precision@k"]:
        if k in summary:
            lines.append(f"| {k} | {summary[k]:.4f} |")

    lines.append(f"\n### 性能\n")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 平均检索耗时 | {avg_retrieval:.1f}ms |")
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
        "version": "2.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "top_k": EVAL_TOP_K,
            "rerank_enabled": rerank_enabled,
            "rerank_model": settings.RERANK_MODEL if rerank_enabled else None,
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


def auto_compare_with_baseline(summary: dict):
    """自动与基线对比并打印差分"""
    if not BASELINE_PATH.exists():
        print(f"\n  [Baseline] 无基线文件，跳过对比。运行 `python {__file__} --save-baseline` 保存基线。")
        return

    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        base_summary = baseline.get("summary", {})
    except Exception as e:
        print(f"\n  [Baseline] 基线读取失败: {e}")
        return

    diffs = []
    for metric in summary:
        cur = summary[metric]
        base = base_summary.get(metric)
        if base is None:
            continue
        delta = round(cur - base, 4)
        icon = "[+]" if delta > 0.005 else ("[-]" if delta < -0.005 else "[=]")
        diffs.append(f"    {metric:<28s} {base:.4f} → {cur:.4f}  {icon} {delta:+.4f}")

    if diffs:
        print(f"\n  {'─' * 60}")
        print(f"  [Baseline] 与基线对比:")
        for line in diffs:
            print(line)
        print(f"  {'─' * 60}\n")


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

    print(f"\n[Eval] 开始评估（{len(ragas_records)} 条有检索结果）...")
    eval_result = run_ragas_evaluation(records)  # 所有记录都参与评估（自定义指标无限制）

    rerank_enabled = getattr(settings, "RERANK_ENABLED", False)
    print_report(eval_result["summary"], eval_result["detail"], records, rerank_enabled)
    save_markdown(eval_result["summary"], eval_result["detail"], records, rerank_enabled)
    save_json_report(eval_result["summary"], eval_result["detail"], records, rerank_enabled)

    # 与基线自动对比
    auto_compare_with_baseline(eval_result["summary"])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--save-baseline":
        asyncio.run(main())
        save_baseline()
    else:
        asyncio.run(main())
