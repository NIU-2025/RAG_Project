"""
消融实验：寻找 EVAL_THRESHOLD 最优值

背景：
  EVAL_THRESHOLD = 0.38 是评估阶段的语义相似度判定阈值，当前是经验值。

实验方法：
  【重要】所有 embedding 在检索阶段预计算并缓存，评估阶段零 embedding 调用，
          避免 sentence-transformers 重复调用导致的卡死。

  1. 阶段1（检索）：每条查询跑一次完整流水线，预计算并缓存所有 embedding
  2. 阶段2（评估）：对每个候选阈值，只做 numpy 矩阵运算（快，无需 embedding）

输出：
  - 终端打印指标-阈值对照表 + 分析结论
  - JSON 保存到 retrievalVersionReport/abl_reranker_test/abl_eval_threshold_result.json

用法：
    python tests/evaluation/retrievalCode/abl_eval_threshold.py

注意事项：
  - BM25 需要 MySQL 运行中，否则自动降级
  - Reranker 模型首次运行需下载（约 2GB）
"""

import sys
import json
import time
import re
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- 项目路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.vector_store import VectorStore
from app.db.session import AsyncSessionLocal
from app.services.embedding import EmbeddingService
from app.services.reranker import RerankerService
from app.services.retrieval import _bm25_search, _compute_dynamic_weights
from app.core.config import settings

# ============================================================
# 配置
# ============================================================

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
OUTPUT_DIR = Path(__file__).parent.parent / "retrievalVersionReport" / "abl_reranker_test"
OUTPUT_PATH = OUTPUT_DIR / "abl_eval_threshold_result.json"
KB_ID = 1
TOP_K = 5
RERANK_CANDIDATES = 20

# --- 候选阈值范围（步长 0.05，额外包含当前使用的 0.38） ---
CANDIDATE_THRESHOLDS = sorted(set(
    [round(x, 2) for x in np.arange(0.05, 0.95, 0.05)] + [0.38]
))


def load_dataset() -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[数据] 加载测试集: {len(data)} 条 | {DATASET_PATH}")
    return data


# ============================================================
# 工具函数
# ============================================================

def _cosine_sim(a: List[float], b: List[float]) -> float:
    a_norm = np.array(a) / (np.linalg.norm(np.array(a)) + 1e-12)
    b_norm = np.array(b) / (np.linalg.norm(np.array(b)) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def _split_sentences(text: str) -> List[str]:
    """按标点拆句子（与 evaluate_rag.py 保持一致）"""
    sents = [s.strip() for s in re.split(r'[。！？\n；;]', text) if len(s.strip()) > 2]
    return sents if sents else [text]


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)


# ============================================================
# 阶段 1：检索 + embedding 预计算
# ============================================================

async def _retrieve_and_cache(item: dict) -> dict:
    """
    对单条测试数据：
      1. 执行完整检索流水线
      2. 预计算所有需要的 embedding，存入 cache 字段

    返回 dict 包含:
      - contexts, ground_truth, question
      - query_emb: List[float]
      - cached_context_embs: np.ndarray (n_contexts, dim)
      - cached_gt_embs: np.ndarray (n_gt_sents, dim)
      - vec_docs
    """
    query = item["question"]
    gt = item.get("ground_truth", "")

    # ── 1. Query Embedding ──
    query_emb = EmbeddingService.embed_query(query)

    # ── 2. 向量检索 ──
    result = VectorStore.query(
        kb_id=KB_ID,
        query_embedding=query_emb,
        top_k=RERANK_CANDIDATES,
    )
    vec_scores = {}
    vec_docs = {}
    if result["ids"][0]:
        for chroma_id, dist, doc_text in zip(
            result["ids"][0], result["distances"][0], result["documents"][0]
        ):
            sim = 1.0 - dist
            vec_scores[chroma_id] = sim
            vec_docs[chroma_id] = doc_text

    # ── 3. BM25 ──
    bm25_scores = {}
    try:
        async with AsyncSessionLocal() as db:
            bm25_scores = await _bm25_search(KB_ID, query, RERANK_CANDIDATES, db)
    except Exception:
        pass

    # ── 4. 动态权重融合 ──
    vec_w, bm25_w = _compute_dynamic_weights(
        vec_scores=vec_scores,
        bm25_scores=bm25_scores,
        query=query,
        base_vec_w=settings.VECTOR_WEIGHT,
        base_bm25_w=settings.BM25_WEIGHT,
    )
    all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
    combined_scores = {}
    for chroma_id in all_ids:
        v = vec_scores.get(chroma_id, 0.0)
        b = bm25_scores.get(chroma_id, 0.0)
        combined_scores[chroma_id] = v * vec_w + b * bm25_w
    coarse_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

    # ── 5. Reranker 精排 ──
    coarse_candidates = coarse_ids[:RERANK_CANDIDATES]
    candidates = [(cid, vec_docs[cid]) for cid in coarse_candidates if cid in vec_docs and vec_docs[cid]]
    rerank_scores = {}
    if candidates:
        scores = RerankerService.rerank(query, candidates)
        rerank_scores = {cid: rs for (cid, _), rs in zip(candidates, scores)}

    # ── 6. 应用当前 Reranker 降级策略（使用生产环境的 0.1/0.3） ──
    rerank_sorted = sorted(rerank_scores.keys(), key=lambda x: rerank_scores[x], reverse=True)
    top_score = rerank_scores.get(rerank_sorted[0], 0.0) if rerank_sorted else 0.0

    if not rerank_scores or top_score < 0.1:
        final_ids = coarse_ids[:TOP_K]
    elif top_score < 0.3:
        reranked = rerank_sorted[:TOP_K]
        seen = set(reranked)
        filled = list(reranked)
        for cid in coarse_ids:
            if cid not in seen:
                filled.append(cid)
                seen.add(cid)
            if len(filled) >= TOP_K:
                break
        final_ids = filled[:TOP_K]
    else:
        final_ids = rerank_sorted[:TOP_K]

    contexts = [vec_docs.get(cid, "") for cid in final_ids if vec_docs.get(cid, "")]

    # ═══════════════════════════════════════════════
    # 预计算并缓存所有 embedding（关键优化！）
    # ═══════════════════════════════════════════════
    # 1) context embeddings
    valid_contexts = [c for c in contexts if c]
    if valid_contexts:
        c_embs = EmbeddingService.embed_texts(valid_contexts)
        cached_context_embs = _l2_normalize(np.array(c_embs))
    else:
        cached_context_embs = np.zeros((0, 0))

    # 2) ground_truth sentence embeddings
    gt_sents = _split_sentences(gt)
    if gt_sents:
        gt_embs = EmbeddingService.embed_texts(gt_sents)
        cached_gt_embs = _l2_normalize(np.array(gt_embs))
    else:
        cached_gt_embs = np.zeros((0, 0))

    return {
        "question": query,
        "ground_truth": gt,
        "contexts": contexts,
        "query_emb": query_emb,
        "cached_context_embs": cached_context_embs,
        "cached_gt_embs": cached_gt_embs,
        "vec_docs": vec_docs,
    }


# ============================================================
# 阶段 2：指标计算（零 embedding 调用，纯 numpy）
# ============================================================

def _compute_metrics_from_cache(
    cached_gt_embs: np.ndarray,
    cached_context_embs: np.ndarray,
    query_emb: List[float],
    contexts: List[str],
    ground_truth: str,
    threshold: float,
) -> dict:
    """
    基于预缓存的 embedding 矩阵计算所有指标。
    无 embedding 调用，纯 numpy 矩阵运算。
    """
    n_contexts = cached_context_embs.shape[0]
    n_gt = cached_gt_embs.shape[0]

    # --- retrieval/recall@k & precision@k ---
    if n_contexts > 0 and n_gt > 0:
        sim = np.dot(cached_gt_embs, cached_context_embs.T)  # (n_gt, n_contexts)
        covered = np.max(sim, axis=1) >= threshold
        context_rel = np.max(sim, axis=0) >= threshold
        ret_recall = float(np.mean(covered)) if n_gt > 0 else 0.0
        ret_precision = float(np.mean(context_rel)) if n_contexts > 0 else 0.0
        # 收集分布
        max_per_gt = np.max(sim, axis=1).tolist() if n_gt > 0 else []
        all_pairs = sim.flatten().tolist()
    else:
        ret_recall = 0.0
        ret_precision = 0.0
        max_per_gt = []
        all_pairs = []

    # --- context_precision ---
    if n_contexts > 0 and query_emb is not None:
        q_norm = np.array(query_emb) / (np.linalg.norm(np.array(query_emb)) + 1e-12)
        ctx_sims = np.dot(cached_context_embs, q_norm)  # (n_contexts,)
        relevant = int(np.sum(ctx_sims >= threshold))
        ctx_precision = round(relevant / n_contexts, 4)
    else:
        ctx_precision = 0.0

    # --- context_recall ---
    ctx_recall = round(ret_recall, 4)  # 逻辑相同

    # --- info_point_coverage ---
    if n_gt > 0 and n_contexts > 0:
        sim2 = np.dot(cached_gt_embs, cached_context_embs.T)
        covered_ip = np.max(sim2, axis=1) >= threshold
        ip_coverage = round(float(np.mean(covered_ip)), 4) if n_gt > 0 else 0.0
    else:
        ip_coverage = 0.0

    # --- faithfulness (embedding 版) ---
    faithfulness = round(ret_recall, 4)  # 逻辑相同

    # --- f1 ---
    f1_ret = round(
        2 * ret_recall * ret_precision / (ret_recall + ret_precision + 1e-8), 4
    ) if (ret_recall + ret_precision) > 0 else 0.0

    return {
        "retrieval_recall": round(ret_recall, 4),
        "retrieval_precision": round(ret_precision, 4),
        "context_precision": ctx_precision,
        "context_recall": ctx_recall,
        "info_point_coverage": ip_coverage,
        "faithfulness_embedding": faithfulness,
        "f1_retrieval": f1_ret,
        "gt_max_sims": max_per_gt,
        "all_pair_sims": all_pairs,
    }


# ============================================================
# 主实验
# ============================================================

async def run_experiment():
    dataset = load_dataset()
    n_queries = len(dataset)

    # ── 阶段 1：检索 + 预计算 embedding ──
    print("\n[阶段 1/2] 执行检索流水线 + 预计算 embedding 缓存...")
    cached = []
    for idx, item in enumerate(dataset):
        print(f"  [{idx+1}/{n_queries}] {item['question'][:40]}...")
        c = await _retrieve_and_cache(item)
        cached.append(c)

    # ── 收集所有 gt_max_sims / all_pair_sims ──
    all_gt_max_sims = []
    all_pair_sims = []
    for c in cached:
        # 对每个缓存结果算一次 sim 矩阵用于分布分析
        nc = c["cached_context_embs"].shape[0]
        ng = c["cached_gt_embs"].shape[0]
        if nc > 0 and ng > 0:
            sim = np.dot(c["cached_gt_embs"], c["cached_context_embs"].T)
            all_gt_max_sims.extend(np.max(sim, axis=1).tolist())
            all_pair_sims.extend(sim.flatten().tolist())

    # ── 阶段 2：纯 numpy 评估（零 embedding！） ──
    print(f"\n[阶段 2/2] 评估 {len(CANDIDATE_THRESHOLDS)} 个阈值（零 embedding 调用）...")
    threshold_metrics = {}

    for idx, th in enumerate(CANDIDATE_THRESHOLDS):
        metrics_agg = {
            "retrieval_recall": [], "retrieval_precision": [],
            "context_precision": [], "context_recall": [],
            "info_point_coverage": [], "faithfulness_embedding": [],
        }

        for c in cached:
            m = _compute_metrics_from_cache(
                cached_gt_embs=c["cached_gt_embs"],
                cached_context_embs=c["cached_context_embs"],
                query_emb=c["query_emb"],
                contexts=c["contexts"],
                ground_truth=c["ground_truth"],
                threshold=th,
            )
            metrics_agg["retrieval_recall"].append(m["retrieval_recall"])
            metrics_agg["retrieval_precision"].append(m["retrieval_precision"])
            metrics_agg["context_precision"].append(m["context_precision"])
            metrics_agg["context_recall"].append(m["context_recall"])
            metrics_agg["info_point_coverage"].append(m["info_point_coverage"])
            metrics_agg["faithfulness_embedding"].append(m["faithfulness_embedding"])

        threshold_metrics[th] = {
            "retrieval_recall": round(float(np.mean(metrics_agg["retrieval_recall"])), 4),
            "retrieval_precision": round(float(np.mean(metrics_agg["retrieval_precision"])), 4),
            "context_precision": round(float(np.mean(metrics_agg["context_precision"])), 4),
            "context_recall": round(float(np.mean(metrics_agg["context_recall"])), 4),
            "info_point_coverage": round(float(np.mean(metrics_agg["info_point_coverage"])), 4),
            "faithfulness_embedding": round(float(np.mean(metrics_agg["faithfulness_embedding"])), 4),
            "f1_retrieval": round(
                2 * float(np.mean(metrics_agg["retrieval_recall"])) * float(np.mean(metrics_agg["retrieval_precision"]))
                / (float(np.mean(metrics_agg["retrieval_recall"])) + float(np.mean(metrics_agg["retrieval_precision"])) + 1e-8),
                4,
            ),
        }

        if (idx + 1) % 5 == 0:
            print(f"  进度: {idx+1}/{len(CANDIDATE_THRESHOLDS)}")

    # ============================================================
    # 输出结果
    # ============================================================

    print(f"\n{'='*90}")
    print(f"  EVAL_THRESHOLD 消融实验结果")
    print(f"  数据集: {n_queries} 条 | TOP_K={TOP_K}")
    print(f"{'='*90}")
    print(f"  {'阈值':>6s} {'Ret-Recall':>10s} {'Ret-Prec':>10s} {'Ctx-Prec':>10s} {'Ctx-Recall':>10s} {'IPCov':>7s} {'Faith':>7s} {'F1-Ret':>8s}")
    print(f"  {'-'*76}")

    output_rows = []
    for th in CANDIDATE_THRESHOLDS:
        m = threshold_metrics[th]
        note = " ← 当前值" if abs(th - 0.38) < 0.001 else ""
        output_rows.append({"threshold": th, **m})
        print(
            f"  {th:>6.2f}  {m['retrieval_recall']:>10.4f}  {m['retrieval_precision']:>10.4f}  "
            f"{m['context_precision']:>10.4f}  {m['context_recall']:>10.4f}  {m['info_point_coverage']:>7.4f}  "
            f"{m['faithfulness_embedding']:>7.4f}  {m['f1_retrieval']:>8.4f}{note}"
        )

    print(f"{'='*90}")

    # ============================================================
    # 分析
    # ============================================================

    print(f"\n  [分析] 指标稳定区间")
    print(f"{'─'*60}")

    high_recall = [r for r in output_rows if r["retrieval_recall"] >= 0.95]
    if high_recall:
        best_p = max(high_recall, key=lambda x: x["retrieval_precision"])
        print(f"    高召回区间 (Recall ≥ 0.95): {high_recall[0]['threshold']:.2f} ~ {high_recall[-1]['threshold']:.2f}")
        print(f"    该区间内最优: {best_p['threshold']:.2f}  "
              f"(Recall={best_p['retrieval_recall']:.4f}, Precision={best_p['retrieval_precision']:.4f}, "
              f"F1={best_p['f1_retrieval']:.4f})")

    best_f1 = max(output_rows, key=lambda x: x["f1_retrieval"])
    print(f"    检索 F1 最优: {best_f1['threshold']:.2f} (F1={best_f1['f1_retrieval']:.4f})")

    if all_pair_sims and all_gt_max_sims:
        print(f"\n  [分析] 相似度分布特征")
        print(f"{'─'*60}")
        print(f"    gt↔context 全部配对: N={len(all_pair_sims)}, "
              f"mean={np.mean(all_pair_sims):.4f}, median={np.median(all_pair_sims):.4f}")
        print(f"    gt 最佳匹配: N={len(all_gt_max_sims)}, "
              f"mean={np.mean(all_gt_max_sims):.4f}, median={np.median(all_gt_max_sims):.4f}, "
              f"p25={np.percentile(all_gt_max_sims, 25):.4f}, p75={np.percentile(all_gt_max_sims, 75):.4f}")

    # 推荐
    print(f"\n  [推荐]")
    current_row = next(r for r in output_rows if abs(r["threshold"] - 0.38) < 0.001)
    print(f"    当前值 0.38:  Recall={current_row['retrieval_recall']:.4f}, "
          f"Precision={current_row['retrieval_precision']:.4f}, F1={current_row['f1_retrieval']:.4f}")

    if abs(best_f1["threshold"] - 0.38) > 0.01:
        print(f"    推荐值 {best_f1['threshold']:.2f}:  Recall={best_f1['retrieval_recall']:.4f}, "
              f"Precision={best_f1['retrieval_precision']:.4f}, F1={best_f1['f1_retrieval']:.4f}")
        print(f"    → 若需修改: evaluate_rag.py#L83  EVAL_THRESHOLD = {best_f1['threshold']:.2f}")

    # --- 保存结果 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "top_k": TOP_K,
            "n_queries": n_queries,
            "candidate_thresholds": CANDIDATE_THRESHOLDS,
            "dataset": str(DATASET_PATH),
        },
        "results": output_rows,
        "distribution": {
            "gt_max_sim_mean": round(float(np.mean(all_gt_max_sims)), 4) if all_gt_max_sims else None,
            "gt_max_sim_median": round(float(np.median(all_gt_max_sims)), 4) if all_gt_max_sims else None,
            "gt_max_sim_p25": round(float(np.percentile(all_gt_max_sims, 25)), 4) if all_gt_max_sims else None,
            "gt_max_sim_p75": round(float(np.percentile(all_gt_max_sims, 75)), 4) if all_gt_max_sims else None,
            "all_pair_sim_mean": round(float(np.mean(all_pair_sims)), 4) if all_pair_sims else None,
            "all_pair_sim_median": round(float(np.median(all_pair_sims)), 4) if all_pair_sims else None,
        },
        "recommendation": {
            "current_value": 0.38,
            "recommended_by_f1": best_f1["threshold"],
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  [保存] 结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
