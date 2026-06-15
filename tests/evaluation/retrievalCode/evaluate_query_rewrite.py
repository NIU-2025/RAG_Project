"""
Query 改写 A/B 测试脚本 — 对比改写前后对检索质量的影响

测试方式:
  对 40 条多轮对话追问，每条分别走两条链路：
    Group A (对照组/不改写): follow_up → 直接检索
    Group B (实验组/改写后): rewrite_query_if_needed() → 检索
  用 ground_truth 与检索结果的 embedding 相似度评估 Recall@K / Precision@K / MRR

用法:
    cd d:/AI_code/RAGProject
    python tests/evaluation/retrievalCode/evaluate_query_rewrite.py

输出目录:
    tests/evaluation/retrievalVersionReport/QueryReport/
      - query_report_group_a_no_rewrite.md    (对照组报告)
      - query_report_group_b_with_rewrite.md   (实验组报告)
      - query_ab_comparison_report.md          (对比报告)
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
# venv_pkgs 放后面，避免覆盖系统已安装的同名包（如 numpy）
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "venv_pkgs"))

from dotenv import load_dotenv
load_dotenv()

hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home
hf_endpoint = os.getenv("HF_ENDPOINT")
if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint

import numpy as np

from app.core.config import settings

# A/B 测试不需要 reranker（两侧相同），跳过 FlagEmbedding 依赖
settings.RERANK_ENABLED = False

from app.db.session import AsyncSessionLocal
from app.services.retrieval import RetrievalService
from app.services.embedding import EmbeddingService
from app.services.query_rewriter import rewrite_query_if_needed

# ── 路径 ──
DATASET_PATH = Path(__file__).parent / "query_ab_test_dataset.json"
REPORT_DIR = Path(__file__).parent.parent / "retrievalVersionReport" / "QueryReport"
KB_ID = 1
TOP_K = 5
EVAL_THRESHOLD = 0.40  # embedding 相似度阈值


def _cosine_sim(a: List[float], b: List[float]) -> float:
    a_norm = np.array(a) / (np.linalg.norm(a) + 1e-12)
    b_norm = np.array(b) / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm))


def _compute_retrieval_metrics_single(query: str, contexts: List[str], ground_truth: str) -> dict:
    """
    计算单条检索的指标。
    用 ground_truth 拆句后与 contexts 做 embedding 相似度判断。
    """
    if not ground_truth or not contexts or all(not c for c in contexts):
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "first_hit_rank": 0}

    gt_sents = [s.strip() for s in re.split(r'[。！？\n;；]', ground_truth) if len(s.strip()) > 2]
    if not gt_sents:
        gt_sents = [ground_truth]

    gt_embs = EmbeddingService.embed_texts(gt_sents)
    c_embs = EmbeddingService.embed_texts([c for c in contexts if c])
    if not gt_embs or not c_embs:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "first_hit_rank": 0}

    gt_arr = np.array(gt_embs)
    c_arr = np.array(c_embs)
    gt_norm = gt_arr / (np.linalg.norm(gt_arr, axis=1, keepdims=True) + 1e-12)
    c_norm = c_arr / (np.linalg.norm(c_arr, axis=1, keepdims=True) + 1e-12)

    sim = np.dot(gt_norm, c_norm.T)  # (n_gt, n_contexts)
    covered = np.max(sim, axis=1) >= EVAL_THRESHOLD  # 哪些 gt 句子被覆盖
    context_rel = np.max(sim, axis=0) >= EVAL_THRESHOLD  # 哪些 context 是相关的

    recall = float(np.mean(covered)) if len(covered) > 0 else 0.0
    precision = float(np.mean(context_rel)) if len(context_rel) > 0 else 0.0

    # MRR: 第一个相关 context 的排名倒数
    first_hit_rank = 0
    for i, rel in enumerate(context_rel):
        if rel:
            first_hit_rank = i + 1
            break
    mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "mrr": round(mrr, 4),
        "first_hit_rank": first_hit_rank,
    }


async def run_single_search(query: str) -> dict:
    """单次检索，返回上下文列表和耗时"""
    t0 = time.perf_counter()
    async with AsyncSessionLocal() as db:
        results = await RetrievalService.search(
            kb_id=KB_ID,
            query=query,
            top_k=TOP_K,
            db=db,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    contexts = [r.content for r in results] if results else []
    return {
        "contexts": contexts,
        "result_count": len(results),
        "elapsed_ms": round(elapsed_ms, 1),
    }


async def evaluate_group(
    dataset: List[dict],
    group_name: str,
    use_rewrite: bool,
) -> List[dict]:
    """评估一组（改写或不改写）"""
    records = []
    for i, item in enumerate(dataset):
        qid = item["id"]
        follow_up = item["follow_up"]
        history = item["history"]
        ground_truth = item["ground_truth"]

        # ── 改写阶段 ──
        rewrite_ms = 0.0
        rewrite_success = False
        rewritten = follow_up

        if use_rewrite:
            t0 = time.perf_counter()
            try:
                rewritten = await rewrite_query_if_needed(
                    current_query=follow_up,
                    history=history,
                    provider=settings.DEFAULT_LLM_PROVIDER,
                    db=None,
                )
                rewrite_ms = (time.perf_counter() - t0) * 1000
                rewrite_success = (rewritten != follow_up)
            except Exception as e:
                rewrite_ms = (time.perf_counter() - t0) * 1000
                print(f"  [WARN] Q{qid} 改写失败: {e}，使用原 query")
                rewritten = follow_up

        # ── 检索阶段 ──
        search_result = await run_single_search(rewritten)

        # ── 计算指标 ──
        metrics = _compute_retrieval_metrics_single(
            query=rewritten,
            contexts=search_result["contexts"],
            ground_truth=ground_truth,
        )

        record = {
            "id": qid,
            "category": item["category"],
            "follow_up": follow_up,
            "rewritten": rewritten,
            "rewrite_success": rewrite_success,
            "rewrite_ms": round(rewrite_ms, 1),
            "retrieval_ms": search_result["elapsed_ms"],
            "total_ms": round(rewrite_ms + search_result["elapsed_ms"], 1),
            "result_count": search_result["result_count"],
            "contexts": search_result["contexts"],
            "ground_truth": ground_truth,
            **metrics,
        }
        records.append(record)

        status = "✓改写" if rewrite_success else ("✗未改写" if use_rewrite else "─")
        print(f"  [{group_name}] Q{qid:2d} {status} | "
              f"follow_up={follow_up[:25]:25s} → rewritten={rewritten[:30]:30s} | "
              f"R@{TOP_K}={metrics['recall']:.2f} MRR={metrics['mrr']:.2f}")

    return records


def compute_summary(records: List[dict]) -> dict:
    """计算一组记录的汇总指标"""
    n = len(records)
    if n == 0:
        return {}

    recall = np.mean([r["recall"] for r in records])
    precision = np.mean([r["precision"] for r in records])
    mrr = np.mean([r["mrr"] for r in records])
    rewrite_rate = np.mean([1.0 if r["rewrite_success"] else 0.0 for r in records])
    avg_rewrite_ms = np.mean([r["rewrite_ms"] for r in records])
    avg_retrieval_ms = np.mean([r["retrieval_ms"] for r in records])
    avg_total_ms = np.mean([r["total_ms"] for r in records])
    hit_rate = np.mean([1.0 if r["first_hit_rank"] > 0 else 0.0 for r in records])

    # 按类别汇总
    categories = {}
    for r in records:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"recalls": [], "mrrs": []}
        categories[cat]["recalls"].append(r["recall"])
        categories[cat]["mrrs"].append(r["mrr"])

    cat_summary = {}
    for cat, vals in categories.items():
        cat_summary[cat] = {
            "count": len(vals["recalls"]),
            "avg_recall": round(float(np.mean(vals["recalls"])), 4),
            "avg_mrr": round(float(np.mean(vals["mrrs"])), 4),
        }

    return {
        "count": n,
        "recall_at_k": round(float(recall), 4),
        "precision_at_k": round(float(precision), 4),
        "mrr": round(float(mrr), 4),
        "hit_rate": round(float(hit_rate), 4),
        "rewrite_rate": round(float(rewrite_rate), 4),
        "avg_rewrite_ms": round(float(avg_rewrite_ms), 1),
        "avg_retrieval_ms": round(float(avg_retrieval_ms), 1),
        "avg_total_ms": round(float(avg_total_ms), 1),
        "by_category": cat_summary,
    }


def generate_group_report(records: List[dict], summary: dict, group_name: str, group_label: str) -> str:
    """生成单组报告 Markdown"""
    lines = []
    lines.append(f"# Query 改写 A/B 测试 — {group_label}")
    lines.append("")
    lines.append(f"> 测试集: {summary['count']} 条多轮追问 | 知识库: KB_ID={KB_ID} | Top-K={TOP_K}")
    lines.append(f"> Embedding 模型: {settings.EMBEDDING_MODEL} | 相似度阈值: {EVAL_THRESHOLD}")
    lines.append(f"> 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    lines.append("## 综合指标")
    lines.append("")
    lines.append("| 指标 | 得分 |")
    lines.append("|------|------|")
    lines.append(f"| recall@{TOP_K} | {summary['recall_at_k']} |")
    lines.append(f"| precision@{TOP_K} | {summary['precision_at_k']} |")
    lines.append(f"| MRR | {summary['mrr']} |")
    lines.append(f"| Hit Rate (有命中的比例) | {summary['hit_rate']} |")
    lines.append(f"| 改写触发率 | {summary['rewrite_rate']} |")
    lines.append(f"| 平均改写耗时 | {summary['avg_rewrite_ms']}ms |")
    lines.append(f"| 平均检索耗时 | {summary['avg_retrieval_ms']}ms |")
    lines.append(f"| 平均端到端耗时 | {summary['avg_total_ms']}ms |")
    lines.append("")

    # 按类别
    lines.append("## 按类别分析")
    lines.append("")
    lines.append("| 类别 | 条数 | Recall@K | MRR |")
    lines.append("|------|------|----------|-----|")
    for cat, vals in sorted(summary["by_category"].items()):
        lines.append(f"| {cat} | {vals['count']} | {vals['avg_recall']} | {vals['avg_mrr']} |")
    lines.append("")

    # 详细记录
    lines.append("## 逐条记录")
    lines.append("")
    lines.append("| ID | 类别 | 追问 | 改写后 | 改写? | Recall@K | MRR | 首位命中 | 检索耗时 |")
    lines.append("|----|------|------|--------|-------|----------|-----|----------|----------|")
    for r in records:
        fwd_short = r["follow_up"][:30]
        rew_short = r["rewritten"][:40] if r["rewritten"] != r["follow_up"] else "─"
        rew_flag = "✓" if r["rewrite_success"] else "─"
        hit_label = f"#{r['first_hit_rank']}" if r["first_hit_rank"] > 0 else "✗"
        lines.append(f"| {r['id']} | {r['category']} | {fwd_short} | {rew_short} | {rew_flag} | "
                      f"{r['recall']} | {r['mrr']} | {hit_label} | {r['retrieval_ms']}ms |")
    lines.append("")

    return "\n".join(lines)


def generate_comparison_report(summary_a: dict, summary_b: dict, records_a: List[dict], records_b: List[dict]) -> str:
    """生成 A/B 对比报告"""
    lines = []
    lines.append("# Query 改写 A/B 对比报告")
    lines.append("")
    lines.append(f"> Group A (对照组): 不改写，直接用追问检索")
    lines.append(f"> Group B (实验组): 改写后再检索")
    lines.append(f"> 测试集: {summary_a['count']} 条多轮追问 | Top-K={TOP_K}")
    lines.append(f"> 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 整体对比
    lines.append("## 整体指标对比")
    lines.append("")
    lines.append("| 指标 | Group A (不改写) | Group B (改写后) | 提升(△) |")
    lines.append("|------|-----------------|-----------------|---------|")

    delta_recall = round(summary_b["recall_at_k"] - summary_a["recall_at_k"], 4)
    delta_precision = round(summary_b["precision_at_k"] - summary_a["precision_at_k"], 4)
    delta_mrr = round(summary_b["mrr"] - summary_a["mrr"], 4)
    delta_hit = round(summary_b["hit_rate"] - summary_a["hit_rate"], 4)

    def fmt_delta(d):
        if d > 0:
            return f"**+{d:.2%}**" if d > 0.05 else f"+{d:.2%}"
        elif d < 0:
            return f"{d:.2%}"
        else:
            return "—"

    lines.append(f"| recall@{TOP_K} | {summary_a['recall_at_k']} | {summary_b['recall_at_k']} | {fmt_delta(delta_recall)} |")
    lines.append(f"| precision@{TOP_K} | {summary_a['precision_at_k']} | {summary_b['precision_at_k']} | {fmt_delta(delta_precision)} |")
    lines.append(f"| MRR | {summary_a['mrr']} | {summary_b['mrr']} | {fmt_delta(delta_mrr)} |")
    lines.append(f"| Hit Rate | {summary_a['hit_rate']} | {summary_b['hit_rate']} | {fmt_delta(delta_hit)} |")
    lines.append(f"| 平均检索耗时 | {summary_a['avg_retrieval_ms']}ms | {summary_b['avg_retrieval_ms']}ms | {summary_b['avg_retrieval_ms']-summary_a['avg_retrieval_ms']:+.1f}ms |")
    lines.append(f"| 平均改写耗时 | — | {summary_b['avg_rewrite_ms']}ms | — |")
    lines.append(f"| 平均端到端耗时 | {summary_a['avg_total_ms']}ms | {summary_b['avg_total_ms']}ms | {summary_b['avg_total_ms']-summary_a['avg_total_ms']:+.1f}ms |")
    lines.append("")

    # 按类别对比
    lines.append("## 按类别对比")
    lines.append("")
    all_cats = sorted(set(summary_a["by_category"].keys()) | set(summary_b["by_category"].keys()))
    lines.append("| 类别 | GroupA Recall@K | GroupB Recall@K | △Recall | GroupA MRR | GroupB MRR | △MRR |")
    lines.append("|------|----------------|----------------|---------|------------|------------|-------|")
    for cat in all_cats:
        ca = summary_a["by_category"].get(cat, {})
        cb = summary_b["by_category"].get(cat, {})
        ra = ca.get("avg_recall", "—")
        rb = cb.get("avg_recall", "—")
        ma = ca.get("avg_mrr", "—")
        mb = cb.get("avg_mrr", "—")
        dr = fmt_delta(rb - ra) if isinstance(ra, float) and isinstance(rb, float) else "—"
        dm = fmt_delta(mb - ma) if isinstance(ma, float) and isinstance(mb, float) else "—"
        lines.append(f"| {cat} | {ra} | {rb} | {dr} | {ma} | {mb} | {dm} |")
    lines.append("")

    # 改写成功率
    lines.append("## 改写效果分析")
    lines.append("")
    rewrite_count = sum(1 for r in records_b if r["rewrite_success"])
    lines.append(f"- 改写触发率: {summary_b['rewrite_rate']:.1%} ({rewrite_count}/{summary_b['count']})")
    lines.append(f"- 改写平均耗时: {summary_b['avg_rewrite_ms']}ms")
    lines.append(f"- 改写带来检索耗时变化: {summary_b['avg_retrieval_ms'] - summary_a['avg_retrieval_ms']:+.1f}ms")
    lines.append("")

    # 逐条对比
    lines.append("## 逐条对比 (Recall 变化排序)")
    lines.append("")
    diffs = []
    for ra, rb in zip(records_a, records_b):
        delta = rb["recall"] - ra["recall"]
        diffs.append((ra["id"], ra["category"], ra["follow_up"], rb["rewritten"],
                       ra["recall"], rb["recall"], delta,
                       ra["mrr"], rb["mrr"]))
    diffs.sort(key=lambda x: x[6], reverse=True)

    lines.append("| ID | 类别 | 追问 | 改写后 | GroupA Recall | GroupB Recall | △Recall | GroupA MRR | GroupB MRR |")
    lines.append("|----|------|------|--------|--------------|--------------|---------|-----------|-----------|")
    for qid, cat, fu, rw, ra_r, rb_r, delta, ra_m, rb_m in diffs:
        fu_short = fu[:25]
        rw_short = rw[:35] if rw != fu else "─"
        delta_str = fmt_delta(delta)
        lines.append(f"| {qid} | {cat} | {fu_short} | {rw_short} | {ra_r} | {rb_r} | {delta_str} | {ra_m} | {rb_m} |")
    lines.append("")

    # 结论
    lines.append("## 结论")
    lines.append("")
    if delta_recall > 0.05:
        lines.append(f"✅ **改写有效**：Recall@{TOP_K} 提升 {delta_recall:.1%}，MRR 提升 {delta_mrr:.1%}，建议继续使用 Query 改写。")
    elif delta_recall > 0:
        lines.append(f"➖ **效果微弱**：Recall@{TOP_K} 仅提升 {delta_recall:.1%}，改写收益不明显，需进一步分析改写质量。")
    else:
        lines.append(f"❌ **改写负收益**：Recall@{TOP_K} 下降 {abs(delta_recall):.1%}，当前改写策略可能引入噪声，建议优化。")

    if delta_mrr > 0.05:
        lines.append(f"- 改写显著提升了相关结果的排名位置（MRR +{delta_mrr:.1%}），有助于改善用户体验。")
    elif delta_mrr > 0:
        lines.append(f"- 改写对排名影响有限（MRR +{delta_mrr:.1%}）。")

    lines.append(f"- 改写带来的额外延迟约 {summary_b['avg_rewrite_ms']}ms，改写触发率 {summary_b['rewrite_rate']:.1%}，整体成本可控。")
    lines.append("")

    return "\n".join(lines)


async def main():
    # ── 1. 加载数据 ──
    print("=" * 60)
    print("  Query 改写 A/B 测试")
    print(f"  数据集: {DATASET_PATH}")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print(f"[ERROR] 数据集不存在: {DATASET_PATH}")
        sys.exit(1)

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    print(f"\n[INFO] 加载测试集: {len(dataset)} 条多轮追问\n")

    # ── 2. 预加载 Embedding 模型 ──
    print("[INFO] 预加载 Embedding 模型...")
    EmbeddingService.preload_model()
    print("[INFO] Embedding 模型加载完成\n")

    # ── 3. Group A: 不改写 ──
    print("─" * 60)
    print("  Group A (对照组): 不改写，直接用追问检索")
    print("─" * 60)
    records_a = await evaluate_group(dataset, "A", use_rewrite=False)
    summary_a = compute_summary(records_a)

    print(f"\n  Group A 汇总:")
    print(f"    Recall@{TOP_K}: {summary_a['recall_at_k']}")
    print(f"    MRR: {summary_a['mrr']}\n")

    # ── 4. Group B: 改写后检索 ──
    print("─" * 60)
    print("  Group B (实验组): 改写后再检索")
    print("─" * 60)
    records_b = await evaluate_group(dataset, "B", use_rewrite=True)
    summary_b = compute_summary(records_b)

    print(f"\n  Group B 汇总:")
    print(f"    Recall@{TOP_K}: {summary_b['recall_at_k']}")
    print(f"    MRR: {summary_b['mrr']}")
    print(f"    改写触发率: {summary_b['rewrite_rate']:.1%}")
    print(f"    平均改写耗时: {summary_b['avg_rewrite_ms']}ms\n")

    # ── 5. 生成报告 ──
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Group A 报告
    report_a = generate_group_report(records_a, summary_a, "A", "Group A (对照组 — 不改写)")
    report_a_path = REPORT_DIR / "query_report_group_a_no_rewrite.md"
    report_a_path.write_text(report_a, encoding="utf-8")
    print(f"[OUTPUT] 对照组报告: {report_a_path}")

    # Group B 报告
    report_b = generate_group_report(records_b, summary_b, "B", "Group B (实验组 — 改写后)")
    report_b_path = REPORT_DIR / "query_report_group_b_with_rewrite.md"
    report_b_path.write_text(report_b, encoding="utf-8")
    print(f"[OUTPUT] 实验组报告: {report_b_path}")

    # 对比报告
    report_ab = generate_comparison_report(summary_a, summary_b, records_a, records_b)
    report_ab_path = REPORT_DIR / "query_ab_comparison_report.md"
    report_ab_path.write_text(report_ab, encoding="utf-8")
    print(f"[OUTPUT] 对比报告: {report_ab_path}")

    # ── 6. 终端输出简要对比 ──
    print("\n" + "=" * 60)
    print("  A/B 测试完成 — 简要结果")
    print("=" * 60)
    print(f"  {'指标':<20s} {'Group A(不改写)':<18s} {'Group B(改写后)':<18s} {'△提升':<10s}")
    print(f"  {'-'*66}")
    delta_r = summary_b["recall_at_k"] - summary_a["recall_at_k"]
    delta_p = summary_b["precision_at_k"] - summary_a["precision_at_k"]
    delta_m = summary_b["mrr"] - summary_a["mrr"]
    print(f"  {'Recall@' + str(TOP_K):<20s} {summary_a['recall_at_k']:<18} {summary_b['recall_at_k']:<18} {delta_r:>+.2%}")
    print(f"  {'MRR':<20s} {summary_a['mrr']:<18} {summary_b['mrr']:<18} {delta_m:>+.2%}")
    print(f"  {'改写触发率':<20s} {'—':<18s} {summary_b['rewrite_rate']:<18.1%} {'—':<10s}")
    print(f"  {'改写耗时':<20s} {'—':<18s} {summary_b['avg_rewrite_ms']:<18.1f}ms{'—':<10s}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
