"""
消融实验：寻找 Reranker 最佳降级阈值（三层降级策略）

背景：
  当前硬编码的降级阈值是 top_score < 0.1 → 回退粗排，< 0.3 → 混合。
  这些值是经验值，未在你的数据集上验证。

实验方法：
  1. 对每条测试查询，执行完整流水线：向量检索 → BM25检索 → 动态权重融合 → Reranker精排
  2. 对同一组 Reranker 分数，离线回放（应用）不同阈值组合的降级逻辑
  3. 粗排分数使用 combined_scores（向量×权重 + BM25×权重），与生产逻辑一致
  4. 计算每种阈值组合的 Recall@K / Precision@K，输出对比表

输出：
  - 终端打印对比表 + TOP5 推荐
  - JSON 结果保存到 retrievalVersionReport/abl_reranker_test/abl_reranker_thresholds_result_v1.json

用法：
    python tests/evaluation/retrievalCode/abl_reranker_thresholds.py

注意事项：
  - BM25 需要 MySQL 运行中。若 MySQL 不可用，自动降级为纯向量+重排（打印 WARNING）
  - Reranker 模型首次运行会自动下载（约 2GB），请耐心等待
"""

import sys
import json
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# --- 项目路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.vector_store import VectorStore
from app.db.session import AsyncSessionLocal
from app.services.embedding import EmbeddingService
from app.services.reranker import RerankerService
from app.services.retrieval import _bm25_search, _compute_dynamic_weights
from app.core.config import settings
from app.core.logger import logger

# ============================================================
# 配置
# ============================================================

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
OUTPUT_DIR = Path(__file__).parent.parent / "retrievalVersionReport" / "abl_reranker_test"
OUTPUT_PATH = OUTPUT_DIR / "abl_reranker_thresholds_result_v1.json"
KB_ID = 1
TOP_K = 5                      # 最终返回条数
RERANK_CANDIDATES = 20         # 送入重排器的候选数（= TOP_K × RERANK_MULTIPLIER，默认 5×4）
EVAL_THRESHOLD = 0.38          # 语义相关判定阈值（复用评估脚本的值）

# --- 待测试的降级阈值组合 ---
# 每组: (low_threshold, high_threshold)
#   top_score < low_th  → 回退粗排（不用重排结果）
#   top_score < high_th → 混合（重排 + 粗排补充）
#   top_score >= high_th → 全用重排结果
THRESHOLD_PAIRS = [
    # (low, high)
    (0.00, 0.00),   # 基线：永远全用重排（从不降级）
    (0.00, 0.10),   # 极宽松
    (0.00, 0.20),
    (0.00, 0.30),
    (0.00, 0.50),
    (0.05, 0.15),
    (0.05, 0.20),
    (0.05, 0.25),
    (0.05, 0.30),
    (0.10, 0.20),
    (0.10, 0.30),   # ◄ 当前默认值
    (0.10, 0.40),
    (0.15, 0.25),
    (0.15, 0.30),
    (0.15, 0.35),
    (0.15, 0.45),
    (0.20, 0.30),
    (0.20, 0.40),
    (0.20, 0.50),
    (0.25, 0.35),
    (0.25, 0.45),
    (0.30, 0.40),
    (0.30, 0.50),
    (0.10, 0.10),   # 特例：low=high，即要么回退要么全用，不混合
    (0.20, 0.20),
    (0.30, 0.30),
    (0.05, 0.05),
    (0.00, 0.05),   # 极严格：几乎永远混合或全用重排
]

# ============================================================
# 数据加载
# ============================================================

def load_dataset() -> List[dict]:
    """加载评估数据集"""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[数据] 加载测试集: {len(data)} 条 | {DATASET_PATH}")
    return data


# ============================================================
# 检索流水线（忠实复现 production 逻辑）
# ============================================================

async def _get_query_embedding(query: str) -> List[float]:
    """获取查询向量"""
    return EmbeddingService.embed_query(query)


def _vector_search(query_emb: List[float], top_k: int) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    向量检索：返回 (vec_scores, vec_docs)
      - vec_scores: {chroma_id: cosine_similarity}
      - vec_docs:   {chroma_id: {"text": ..., "metadata": ...}}
    """
    result = VectorStore.query(
        kb_id=KB_ID,
        query_embedding=query_emb,
        top_k=top_k,
    )
    if not result["ids"][0]:
        return {}, {}

    ids = result["ids"][0]
    distances = result["distances"][0]
    documents = result["documents"][0]

    vec_scores = {}
    vec_docs = {}
    for chroma_id, dist, doc_text in zip(ids, distances, documents):
        sim = 1.0 - dist
        vec_scores[chroma_id] = sim
        vec_docs[chroma_id] = {"text": doc_text, "metadata": {}}

    return vec_scores, vec_docs


def _rerank(query: str, candidates: List[Tuple[str, str]]) -> Dict[str, float]:
    """调用 Reranker，返回 {chroma_id: score}"""
    if not candidates:
        return {}
    scores = RerankerService.rerank(query, candidates)
    return {cid: rs for (cid, _), rs in zip(candidates, scores)}


# ============================================================
# 降级逻辑 + 结果构建
# ============================================================

def _apply_threshold(
    low_th: float,
    high_th: float,
    coarse_ids: List[str],
    coarse_scores: Dict[str, float],
    rerank_scores: Dict[str, float],
    vec_docs: Dict[str, str],
    top_k: int,
    score_threshold: float = 0.0,
) -> List[str]:
    """
    应用指定阈值组合的降级逻辑，返回最终选中的 chroma_id 列表。

    参数:
      low_th, high_th    降级阈值
      coarse_ids         粗排候选 ID 列表（按 combined_scores 排序）
      coarse_scores      粗排分数 {id: score}（即 combined_scores）
      rerank_scores      Reranker 分数 {id: score}
      vec_docs           文档文本 {id: text}（纯 fallback 查询用）
      top_k              最终返回 K 条
      score_threshold    粗排最低分过滤

    返回: 选中的 chroma_id 列表（按降级逻辑排序）
    """
    if not rerank_scores:
        # Reranker 未启用 → 回退粗排
        return [cid for cid in coarse_ids[:top_k] if coarse_scores.get(cid, 0) >= score_threshold]

    # 按 Reranker 分数排序
    rerank_sorted = sorted(rerank_scores.keys(), key=lambda x: rerank_scores[x], reverse=True)
    top_score = rerank_scores.get(rerank_sorted[0], 0.0)

    if top_score < low_th:
        # 降级：回退粗排
        return [cid for cid in coarse_ids[:top_k] if coarse_scores.get(cid, 0) >= score_threshold]

    elif top_score < high_th:
        # 混合模式：重排 Top-K + 粗排补充（与 retrieval.py #L187-195 一致）
        reranked = rerank_sorted[:top_k]
        seen = set(reranked)
        filled = list(reranked)
        for cid in coarse_ids:
            if cid not in seen:
                if coarse_scores.get(cid, 0) >= score_threshold:
                    filled.append(cid)
                    seen.add(cid)
            if len(filled) >= top_k:
                break
        return filled[:top_k]

    else:
        # 全用重排结果
        return [cid for cid in rerank_sorted[:top_k]]


# ============================================================
# 指标计算
# ============================================================

def _compute_metrics(
    selected_ids: List[str],
    ground_truth: str,
    vec_docs: Dict[str, str],
) -> Dict[str, float]:
    """
    计算单条查询的 Recall@K 和 Precision@K（与 evaluate_rag.py 保持一致）。
    """
    if not selected_ids or not ground_truth:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0}

    # 拆 ground_truth 句子
    gt_sents = [s.strip() for s in ground_truth.replace("。", "\n").replace("！", "\n")
                .replace("？", "\n").replace("；", "\n").split("\n")
                if len(s.strip()) > 2]
    if not gt_sents:
        gt_sents = [ground_truth]

    # 检索结果文本
    retrieved_texts = [vec_docs.get(cid, "") for cid in selected_ids if vec_docs.get(cid, "")]

    if not gt_sents or not retrieved_texts:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0}

    # Embedding
    gt_embs = EmbeddingService.embed_texts(gt_sents)
    c_embs = EmbeddingService.embed_texts(retrieved_texts)

    if not gt_embs or not c_embs:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0}

    gt_arr = np.array(gt_embs)
    c_arr = np.array(c_embs)
    gt_norm = gt_arr / (np.linalg.norm(gt_arr, axis=1, keepdims=True) + 1e-12)
    c_norm = c_arr / (np.linalg.norm(c_arr, axis=1, keepdims=True) + 1e-12)

    sim = np.dot(gt_norm, c_norm.T)  # (n_gt, n_contexts)

    # Recall: ground_truth 句子被覆盖的比例
    covered = np.max(sim, axis=1) >= EVAL_THRESHOLD
    recall = float(np.mean(covered))

    # Precision: 检索结果中相关比例
    context_rel = np.max(sim, axis=0) >= EVAL_THRESHOLD
    precision = float(np.mean(context_rel))

    return {"recall_at_k": round(recall, 4), "precision_at_k": round(precision, 4)}


# ============================================================
# 主实验逻辑
# ============================================================

async def run_experiment():
    """运行消融实验"""
    dataset = load_dataset()
    n_queries = len(dataset)

    # 统计 BM25 可用性
    bm25_available_count = 0

    # --- 存储每组的汇总结果 ---
    all_results: Dict[str, Dict] = {}
    for low, high in THRESHOLD_PAIRS:
        key = f"({low:.2f}, {high:.2f})"
        all_results[key] = {"recalls": [], "precisions": [], "count": 0}

    # --- 逐条处理 ---
    for idx, item in enumerate(dataset):
        qid = item["id"]
        query = item["question"]
        gt = item.get("ground_truth", "")

        print(f"\n{'='*50}")
        print(f"[{idx+1}/{n_queries}] id={qid} | {query[:50]}")
        print(f"{'='*50}")

        # 1. Query Embedding
        t0 = time.perf_counter()
        query_emb = await _get_query_embedding(query)
        embed_ms = (time.perf_counter() - t0) * 1000
        print(f"  [Embedding] {embed_ms:.0f}ms")

        # 2. 向量检索
        vec_scores, vec_docs = _vector_search(query_emb, RERANK_CANDIDATES)
        if not vec_scores:
            print(f"  [SKIP] 向量检索无结果")
            continue
        print(f"  [Vector] 命中: {len(vec_scores)} 条")

        # 3. BM25 检索（try-catch，失败则降级为纯向量）
        t0 = time.perf_counter()
        bm25_scores = {}
        try:
            async with AsyncSessionLocal() as db:
                bm25_scores = await _bm25_search(KB_ID, query, RERANK_CANDIDATES, db)
        except Exception as e:
            print(f"  [BM25]  跳过（DB 不可用: {e}）")
        bm25_ms = (time.perf_counter() - t0) * 1000

        if bm25_scores:
            bm25_available_count += 1
            print(f"  [BM25]  {bm25_ms:.0f}ms | 命中: {len(bm25_scores)} 条")
        else:
            print(f"  [BM25]  无命中或不可用")

        # 4. 动态权重融合 → combined_scores（与 retrieval.py #L153-157 完全一致）
        vec_w, bm25_w = _compute_dynamic_weights(
            vec_scores=vec_scores,
            bm25_scores=bm25_scores,
            query=query,
            base_vec_w=settings.VECTOR_WEIGHT,
            base_bm25_w=settings.BM25_WEIGHT,
        )
        all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
        combined_scores: Dict[str, float] = {}
        for chroma_id in all_ids:
            vec_s = vec_scores.get(chroma_id, 0.0)
            bm25_s = bm25_scores.get(chroma_id, 0.0)
            combined_scores[chroma_id] = vec_s * vec_w + bm25_s * bm25_w
        coarse_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        print(f"  [Fusion] vec_w={vec_w:.2f} bm25_w={bm25_w:.2f} | 候选: {len(coarse_ids)} 条")

        # 5. 构建 Reranker 候选（取前 RERANK_CANDIDATES 条）
        coarse_candidates = coarse_ids[:RERANK_CANDIDATES]
        candidates = []
        for cid in coarse_candidates:
            # 优先用 vec_docs（含完整文本），否则跳过（BM25-only 且未 fallback）
            info = vec_docs.get(cid)
            if info and info.get("text"):
                candidates.append((cid, info["text"]))

        if not candidates:
            print(f"  [SKIP] 无 Reranker 候选文本")
            continue

        # 6. Reranker 精排（只跑一次，所有阈值复用）
        t0 = time.perf_counter()
        rerank_scores = _rerank(query, candidates)
        rerank_ms = (time.perf_counter() - t0) * 1000
        print(f"  [Reranker] {rerank_ms:.0f}ms | candidates={len(candidates)}")
        if rerank_scores:
            top_score = max(rerank_scores.values())
            print(f"  [Reranker] Top score: {top_score:.4f}")

        # 7. 对每个阈值组合离线回放降级逻辑
        for low, high in THRESHOLD_PAIRS:
            key = f"({low:.2f}, {high:.2f})"

            selected_ids = _apply_threshold(
                low_th=low, high_th=high,
                coarse_ids=coarse_ids,
                coarse_scores=combined_scores,  # ← 使用 combined_scores，与生产一致
                rerank_scores=rerank_scores,
                vec_docs={k: v["text"] for k, v in vec_docs.items()},
                top_k=TOP_K,
                score_threshold=0.0,
            )

            metrics = _compute_metrics(selected_ids, gt, {k: v["text"] for k, v in vec_docs.items()})
            all_results[key]["recalls"].append(metrics["recall_at_k"])
            all_results[key]["precisions"].append(metrics["precision_at_k"])
            all_results[key]["count"] += 1

        # 进度
        if (idx + 1) % 5 == 0:
            print(f"\n  --- 进度: {idx+1}/{n_queries} ---")

    # ================================================================
    # 汇总输出
    # ================================================================

    bm25_status = f"BM25 可用: {bm25_available_count}/{n_queries} 条"
    print(f"\n{'='*70}")
    print(f"  Reranker 降级阈值消融实验结果")
    print(f"  数据集: {n_queries} 条 | TOP_K={TOP_K} | 候选数={RERANK_CANDIDATES} | {bm25_status}")
    print(f"{'='*70}")
    print(f"  {'阈值组合':<20s} {'命中':>4s} {'Recall@K':>10s} {'Precision@K':>12s} {'F1':>8s} {'备注':<20s}")
    print(f"  {'-'*74}")

    output_rows = []
    for low, high in THRESHOLD_PAIRS:
        key = f"({low:.2f}, {high:.2f})"
        data = all_results[key]

        avg_recall = round(float(np.mean(data["recalls"])), 4) if data["recalls"] else 0.0
        avg_precision = round(float(np.mean(data["precisions"])), 4) if data["precisions"] else 0.0
        f1 = round(2 * avg_recall * avg_precision / (avg_recall + avg_precision + 1e-8), 4)

        # 标记
        note = ""
        if abs(low - 0.10) < 0.001 and abs(high - 0.30) < 0.001:
            note = "◄ 当前默认值"
        elif avg_recall >= 0.99:
            note = "✅ 高召回"

        output_rows.append({
            "pair": key,
            "low": low,
            "high": high,
            "count": data["count"],
            "recall": avg_recall,
            "precision": avg_precision,
            "f1": f1,
            "note": note,
        })

        print(f"  {key:<20s} {data['count']:>4d}  {avg_recall:>10.4f}  {avg_precision:>12.4f}  {f1:>8.4f}  {note:<20s}")

    print(f"{'='*70}")

    # --- 排序推荐 ---
    sorted_by_f1 = sorted(output_rows, key=lambda x: x["f1"], reverse=True)
    # 标注 TOP 最优
    if sorted_by_f1:
        best_f1 = sorted_by_f1[0]["f1"]
        for row in output_rows:
            if row["f1"] == best_f1 and not row["note"]:
                row["note"] = "★ 最优 F1"

    # 重新打印一遍带标注的
    print(f"\n  {'阈值组合':<20s} {'命中':>4s} {'Recall@K':>10s} {'Precision@K':>12s} {'F1':>8s} {'备注':<20s}")
    print(f"  {'-'*74}")
    for row in output_rows:
        print(f"  {row['pair']:<20s} {row['count']:>4d}  {row['recall']:>10.4f}  {row['precision']:>12.4f}  {row['f1']:>8.4f}  {row.get('note', ''):<20s}")

    print(f"\n{'='*70}")

    print(f"\n  [推荐] TOP 5 最优阈值组合（按 F1 排序）:")
    print(f"  {'排名':>4s} {'阈值组合':<20s} {'Recall':>8s} {'Precision':>10s} {'F1':>8s}")
    print(f"  {'─'*52}")
    for rank, row in enumerate(sorted_by_f1[:5], 1):
        print(f"  {rank:>4d} {row['pair']:<20s} {row['recall']:>8.4f} {row['precision']:>10.4f} {row['f1']:>8.4f}")

    # --- 保存结果 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({
            "config": {
                "top_k": TOP_K,
                "rerank_candidates": RERANK_CANDIDATES,
                "eval_threshold": EVAL_THRESHOLD,
                "bm25_available": f"{bm25_available_count}/{n_queries}",
                "dataset": str(DATASET_PATH),
            },
            "results": output_rows,
            "recommendation": sorted_by_f1[:5],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  [保存] 结果已保存到: {OUTPUT_PATH}")

    # --- 结论 ---
    best = sorted_by_f1[0]
    current = next(
        (r for r in output_rows if abs(r["low"] - 0.10) < 0.001 and abs(r["high"] - 0.30) < 0.001),
        None,
    )
    print(f"\n  [结论]")
    if current:
        print(f"    当前默认: (0.10, 0.30) → Recall={current['recall']:.4f}, Precision={current['precision']:.4f}, F1={current['f1']:.4f}")
    print(f"    推荐最优: {best['pair']} → Recall={best['recall']:.4f}, Precision={best['precision']:.4f}, F1={best['f1']:.4f}")
    if current:
        print(f"    提升:    Recall {best['recall']-current['recall']:+.4f}, Precision {best['precision']-current['precision']:+.4f}")
    print()
    print(f"  如果需要修改硬编码阈值，请编辑 app/services/retrieval.py:")
    print(f"    第 184 行: if top_score < {best['low']:.2f}:  # 原 0.1")
    print(f"    第 187 行: elif top_score < {best['high']:.2f}:  # 原 0.3")
    print(f"    或通过配置参数化（推荐长期方案）")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_experiment())
