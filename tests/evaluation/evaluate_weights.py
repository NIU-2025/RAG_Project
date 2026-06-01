"""
混合检索权重优化评估脚本
使用 RAGAS ContextPrecision 指标（LLM 语义判断），替代关键词匹配。

用法:
    python tests/evaluation/evaluate_weights.py

输出:
    - 终端打印各权重组合的 ContextPrecision 得分
    - tests/evaluation/weight_eval_report.md 保存详细报告
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.db.vector_store import VectorStore
from app.services.embedding import EmbeddingService
import jieba
from rank_bm25 import BM25Okapi


DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_PATH = Path(__file__).parent / "weight_eval_report.md"
KB_ID = 1


def load_dataset() -> list:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def hybrid_search(
    query: str,
    vec_weight: float,
    bm25_weight: float,
    top_k: int = 5,
    bm25_corpus: Tuple[list, list] = None,
    query_embedding: list = None,
) -> list:
    if query_embedding is None:
        query_embedding = EmbeddingService.embed_query(query)

    vector_results = VectorStore.query(
        kb_id=KB_ID,
        query_embedding=query_embedding,
        top_k=top_k * 4,
    )

    vec_scores = {}
    vec_docs = {}
    if vector_results["ids"] and vector_results["ids"][0]:
        for cid, dist, text, meta in zip(
            vector_results["ids"][0],
            vector_results["distances"][0],
            vector_results["documents"][0],
            vector_results["metadatas"][0],
        ):
            sim = 1.0 - dist
            vec_scores[cid] = sim
            vec_docs[cid] = (text, meta)

    bm25_scores = {}
    if bm25_corpus and bm25_corpus[0]:
        tokenized_corpus, chroma_ids = bm25_corpus
        query_tokens = list(jieba.cut(query))
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)
        max_s = max(scores) if max(scores) > 0 else 1.0
        for i, cid in enumerate(chroma_ids):
            if scores[i] > 0:
                bm25_scores[cid] = scores[i] / max_s

    if len(vec_scores) == 0 and len(bm25_scores) > 0:
        vw, bw = 0.0, 1.0
    elif len(bm25_scores) == 0 and len(vec_scores) > 0:
        vw, bw = 1.0, 0.0
    else:
        vw, bw = vec_weight, bm25_weight

    all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for cid in all_ids:
        combined[cid] = vec_scores.get(cid, 0.0) * vw + bm25_scores.get(cid, 0.0) * bw

    sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:top_k]
    return [(cid, combined[cid], *vec_docs.get(cid, ("", {}))) for cid in sorted_ids]


def run_ragas_eval(records: list) -> dict:
    """用 RAGAS ContextPrecision 指标（LLM Judge）评估所有查询"""
    from ragas import evaluate
    from ragas.metrics import ContextPrecision
    from openai import OpenAI
    from ragas.llms import llm_factory
    from datasets import Dataset

    client = OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )
    judge_llm = llm_factory(
        getattr(settings, "DEEPSEEK_MODEL", "deepseek-chat"),
        client=client,
    )

    ds = Dataset.from_dict({
        "question": [r["question"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    result = evaluate(
        ds,
        metrics=[ContextPrecision()],
        llm=judge_llm,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if "context_precision" in c]
    scores = []
    for col in metric_cols:
        col_mean = df[col].dropna().mean()
        scores.append(round(float(col_mean), 4))

    per_query = []
    for _, row in df.iterrows():
        for col in metric_cols:
            if pd.notna(row.get(col)):
                per_query.append(float(row.get(col, 0)))
                break

    return {
        "context_precision": scores[0] if scores else 0.0,
        "per_query": per_query if per_query else [0.0] * len(records),
    }


async def evaluate_weight_combo(
    vec_w: float,
    bm25_w: float,
    dataset: list,
    bm25_corpus: Tuple[list, list],
    precomputed_embeddings: Dict[str, list],
) -> Dict:
    """评估一组权重配置 — 收集 contexts 后用 RAGAS LLM Judge 打分"""
    records = []
    for item in dataset:
        query = item["question"]
        emb = precomputed_embeddings.get(query)
        results = await hybrid_search(
            query=query,
            vec_weight=vec_w,
            bm25_weight=bm25_w,
            top_k=5,
            bm25_corpus=bm25_corpus,
            query_embedding=emb,
        )
        contexts = [text for _, _, text, _ in results if text]
        records.append({
            "question": query,
            "ground_truth": item["ground_truth"],
            "contexts": contexts if contexts else [" "],
        })

    ragas_result = run_ragas_eval(records)
    return {
        "vec_weight": vec_w,
        "bm25_weight": bm25_w,
        "context_precision": ragas_result["context_precision"],
    }


async def grid_search_weights(dataset: list):
    """网格搜索最优权重组合"""
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        from app.models.db import DocumentChunk
        result = await db.execute(
            select(DocumentChunk.chroma_id, DocumentChunk.content)
            .where(DocumentChunk.kb_id == KB_ID)
        )
        rows = result.all()
        chroma_ids = [r.chroma_id for r in rows]
        tokenized_corpus = [list(jieba.cut(r.content)) for r in rows]
        bm25_corpus = (tokenized_corpus, chroma_ids)

    queries = [item["question"] for item in dataset]
    embeddings = EmbeddingService.embed_texts(queries)
    precomputed = {q: e for q, e in zip(queries, embeddings)}

    weight_combos = []
    for vw in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        bw = round(1.0 - vw, 1)
        weight_combos.append((vw, bw))

    print(f"\n{'='*70}")
    print(f"  混合检索权重评估 — RAGAS ContextPrecision (LLM Judge)")
    print(f"  数据集: {len(dataset)} 条 | 权重组合: {len(weight_combos)} 组")
    print(f"  BM25 语料: {len(chroma_ids)} chunks")
    print(f"{'='*70}\n")

    results = []
    for vw, bw in weight_combos:
        t0 = time.perf_counter()
        res = await evaluate_weight_combo(vw, bw, dataset, bm25_corpus, precomputed)
        elapsed = (time.perf_counter() - t0) * 1000
        res["elapsed_ms"] = elapsed
        results.append(res)
        print(f"  vec={vw:.1f} / bm25={bw:.1f}  →  ContextPrecision={res['context_precision']:.4f}  ({elapsed:.0f}ms)")

    results.sort(key=lambda x: x["context_precision"], reverse=True)
    best = results[0]

    print(f"\n{'='*70}")
    print(f"  最优权重: vec={best['vec_weight']:.1f} / bm25={best['bm25_weight']:.1f}")
    print(f"  ContextPrecision={best['context_precision']:.4f}")
    print(f"{'='*70}\n")

    _generate_report(results, dataset)
    return best


def _generate_report(results: list, dataset: list):
    lines = ["# 混合检索权重评估报告 (RAGAS Judge)\n"]
    lines.append(f"> 数据集: {len(dataset)} 条 | Judge: DeepSeek | 指标: ContextPrecision\n")
    lines.append("| 向量权重 | BM25权重 | ContextPrecision | 耗时(ms) |")
    lines.append("|---------|---------|-----------------|---------|")

    best_cp = max(r["context_precision"] for r in results)
    for r in results:
        marker = " ⭐" if r["context_precision"] == best_cp else ""
        lines.append(
            f"| {r['vec_weight']:.1f} | {r['bm25_weight']:.1f} "
            f"| {r['context_precision']:.4f} | {r['elapsed_ms']:.0f}{marker} |"
        )

    best = results[0]
    lines.append(f"\n## 推荐配置\n")
    lines.append(f"- **向量权重**: {best['vec_weight']:.1f}")
    lines.append(f"- **BM25权重**: {best['bm25_weight']:.1f}")
    lines.append(f"- **ContextPrecision**: {best['context_precision']:.4f}")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"  报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    dataset = load_dataset()
    best = asyncio.run(grid_search_weights(dataset))
