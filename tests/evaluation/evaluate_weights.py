"""
混合检索权重优化评估脚本
使用 embedding 语义相似度替代 LLM Judge，快速稳定不卡死。

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

import numpy as np
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


EMBED_EVAL_THRESHOLD = 0.45  # embedding 相似度阈值


def _cosine_sim(a: list, b: list) -> float:
    """余弦相似度"""
    a_norm = np.array(a) / (np.linalg.norm(a) + 1e-12)
    b_norm = np.array(b) / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def compute_context_precision(records: list) -> dict:
    """
    基于 embedding 语义相似度的 ContextPrecision。
    对每条 query：
      - 计算 query embedding 与每个 context embedding 的余弦相似度
      - 按相似度降序排列（模拟 LLM 的"排序"）
      - ContextPrecision@K = 每个位置 precision 的均值
        precision@i = top-i 中相关 chunk 个数 / i
    """
    per_query = []
    for r in records:
        query = r["question"]
        contexts = r.get("contexts", [])
        if not contexts or all(not c for c in contexts):
            per_query.append(0.0)
            continue

        q_emb = EmbeddingService.embed_query(query)
        c_embs = EmbeddingService.embed_texts([c for c in contexts if c])
        if not c_embs:
            per_query.append(0.0)
            continue

        sims = [_cosine_sim(q_emb, ce) for ce in c_embs]
        # 降序排列 — 模拟 LLM 的"相关排序"
        sims_sorted = sorted(sims, reverse=True)

        # 每步 precision
        relevant_count = 0
        precisions = []
        for i, s in enumerate(sims_sorted, start=1):
            if s >= EMBED_EVAL_THRESHOLD:
                relevant_count += 1
            precisions.append(relevant_count / i)

        per_query.append(float(np.mean(precisions)))

    overall = round(float(np.mean(per_query)), 4) if per_query else 0.0
    return {
        "context_precision": overall,
        "per_query": per_query,
    }


async def evaluate_weight_combo(
    vec_w: float,
    bm25_w: float,
    dataset: list,
    bm25_corpus: Tuple[list, list],
    precomputed_embeddings: Dict[str, list],
) -> Dict:
    """评估一组权重配置 — 收集 contexts 后用 embedding 语义相似度打分（无需 LLM，不卡死）"""
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
            "contexts": contexts if contexts else [],
        })

    cp_result = compute_context_precision(records)
    return {
        "vec_weight": vec_w,
        "bm25_weight": bm25_w,
        "context_precision": cp_result["context_precision"],
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
    print(f"  混合检索权重评估 — Embedding ContextPrecision (本地快速计算)")
    print(f"  数据集: {len(dataset)} 条 | 权重组合: {len(weight_combos)} 组")
    print(f"  BM25 语料: {len(chroma_ids)} chunks | 阈值: {EMBED_EVAL_THRESHOLD}")
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
    lines = ["# 混合检索权重评估报告 (Embedding)\n"]
    lines.append(f"> 数据集: {len(dataset)} 条 | 方法: embedding 余弦相似度 | 阈值: {EMBED_EVAL_THRESHOLD}\n")
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
