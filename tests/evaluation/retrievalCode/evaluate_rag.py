"""
RAGAS 评估脚本
基于 30 条标注测试集，对 RAG 系统进行 4 维度量化评估。

用法:
    cd d:/AI_code/RAGProject
    python tests/evaluation/retrievalCode/evaluate_rag.py --version V1
    python tests/evaluation/retrievalCode/evaluate_rag.py --version V2
    python tests/evaluation/retrievalCode/evaluate_rag.py --version V3

输出:
    - 终端打印评估报告
    - tests/evaluation/retrievalVersionReport/{VERSION}/eval_report.md 保存 Markdown 报告
"""

import asyncio
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

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

# ── 版本预设（控制变量 A/B 测试） ──
VERSION_PRESETS = {
    "V1": {
        "name": "V1-纯向量检索",
        "rerank_enabled": False,
        "vector_weight": 1.0,
        "bm25_weight": 0.0,
        "score_threshold": 0.3,
        "rerank_multiplier": 1,
    },
    "V2": {
        "name": "V2-向量+BM25混合",
        "rerank_enabled": False,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "score_threshold": 0.3,
        "rerank_multiplier": 1,
    },
    "V3": {
        "name": "V3-三级流水线(全量)",
        "rerank_enabled": True,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "score_threshold": 0.3,
        "rerank_multiplier": 4,
    },
}

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
REPORT_BASE_DIR = Path(__file__).parent.parent / "retrievalVersionReport"
KB_ID = 1

EVAL_THRESHOLD = 0.40  # embedding 相似度阈值，用于判断语义相关/忠实

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


# ── LLM Judge 忠实度评估 ──

FAITHFULNESS_JUDGE_PROMPT = """你是一名事实一致性评估专家。你的任务是根据【参考资料】判断【回答】中的每一句话是否被参考资料支持。

评分标准：
- 1.0 = 该句完全被参考资料支持（包括合理推断）
- 0.5 = 该句部分被支持或信息不完整
- 0.0 = 该句没有参考资料支持（编造、与参考资料矛盾）
- 特殊规则："文档中未找到相关信息"、"未找到XX相关内容"这类明确拒绝回答的句子视为忠实，得分为 1.0

请严格按照以下 JSON 格式输出，不要包含其他内容：
{"sentences":["句子1","句子2","句子3"],"sentence_scores":[1.0,0.5,1.0],"score":0.83,"lowest_sentence":"句子2","lowest_score":0.5,"lowest_reason":"该句部分信息未在参考资料中找到"}"""


async def _compute_faithfulness_llm(answer: str, contexts: List[str], question: str = "") -> dict:
    """
    基于 LLM Judge 的忠实度评估。
    调用 LLM（gpt-4o）逐句判断回答是否被上下文支持。
    
    返回格式同 _compute_faithfulness_embedding。
    LLM 调用失败时返回空字典（调用方自行降级）。
    """
    result = {
        "score": 0.0,
        "sentences": [],
        "sentence_scores": [],
        "lowest_sentence": "",
        "lowest_score": 1.0,
        "lowest_reason": "",
    }

    if not answer:
        result["score"] = 1.0
        return result
    if not contexts or all(not c for c in contexts):
        result["score"] = 1.0
        return result

    # 截断过长上下文，避免 token 超限
    context_str = "\n\n".join(contexts)
    if len(context_str) > 12000:
        context_str = context_str[:12000] + "\n\n...（以下内容已截断）"

    user_content = f"【问题】{question}\n\n【参考资料】\n{context_str}\n\n【回答】\n{answer}"

    try:
        llm_response, _ = await asyncio.wait_for(
            LLMService.chat(
                provider="deepseek",
                model=None,  # 使用 settings.DEEPSEEK_MODEL（deepseek-chat）
                messages=[{"role": "system", "content": FAITHFULNESS_JUDGE_PROMPT}],
                user_message=user_content,
                context="",
                db=None,
            ),
            timeout=60.0,
        )
    except asyncio.TimeoutError:
        print(f"  [LLM Judge] 请求超时(>60s)，降级到 embedding")
        return {}
    except Exception as e:
        print(f"  [LLM Judge] 调用失败，降级到 embedding: {e}")
        return {}  # 空字典表示降级

    # 解析 JSON：兼容 LLM 输出带 {{ }} 包裹、```json 代码块等情况
    try:
        json_str = llm_response.strip()

        # 1. 去掉 ```json ... ``` 代码块包裹
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()

        # 2. 去掉 {{ / }} 双重花括号包裹（LLM 有时会转义输出）
        if json_str.startswith("{{") and json_str.endswith("}}"):
            json_str = json_str[1:-1]

        data = json.loads(json_str)

        result["sentences"] = data.get("sentences", [])
        raw_scores = data.get("sentence_scores", [])
        result["sentence_scores"] = []
        for s in raw_scores:
            try:
                val = float(s) if s != "" else 0.0
                result["sentence_scores"].append(max(0.0, min(1.0, val)))
            except (TypeError, ValueError):
                result["sentence_scores"].append(0.0)
        result["score"] = round(float(np.mean(result["sentence_scores"])), 4) if result["sentence_scores"] else 0.0
        result["lowest_sentence"] = data.get("lowest_sentence") or ""
        raw_lowest = data.get("lowest_score")
        try:
            result["lowest_score"] = round(float(raw_lowest), 4) if raw_lowest not in (None, "", "null") else 1.0
        except (TypeError, ValueError):
            result["lowest_score"] = 1.0
        result["lowest_reason"] = data.get("lowest_reason") or ""
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"  [LLM Judge] JSON 解析失败，降级到 embedding: {e}")
        print(f"  [LLM Judge] 原始响应: {llm_response[:200]}")
        return {}

    return result


async def _compute_faithfulness(answer: str, contexts: List[str], threshold: float = None, question: str = "") -> dict:
    """
    忠实度评估入口：优先使用 LLM Judge，失败时降级到 embedding。
    """
    # 优先 LLM Judge
    if threshold is None:  # 仅当非多阈值扫描时使用 LLM
        llm_result = await _compute_faithfulness_llm(answer, contexts, question)
        if llm_result.get("sentences"):  # LLM 成功返回了有效结果
            return llm_result
    # 降级到 embedding
    return _compute_faithfulness_embedding(answer, contexts, threshold)


def _compute_faithfulness_embedding(answer: str, contexts: List[str], threshold: float = None) -> dict:
    """
    基于 embedding 的忠实度。
    将答案拆分为句子，检查每句是否在上下文中找到语义支撑（> 阈值）。
    
    返回：
        {"score": float, "sentences": [str], "sentence_scores": [float], 
         "lowest_sentence": str, "lowest_score": float}
    """
    if threshold is None:
        threshold = EVAL_THRESHOLD

    result = {
        "score": 0.0,
        "sentences": [],
        "sentence_scores": [],
        "lowest_sentence": "",
        "lowest_score": 1.0,
    }

    if not answer:
        return result
    if not contexts or all(not c for c in contexts):
        result["score"] = 1.0
        return result

    sentences = [s.strip() for s in re.split(r'[。！？\n;；]', answer) if len(s.strip()) > 2]
    if not sentences:
        result["score"] = 1.0
        return result

    valid_contexts = [c for c in contexts if c]
    if not valid_contexts:
        result["score"] = 1.0
        return result

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
    window_size = 3
    n = len(max_sims)
    smoothed = np.copy(max_sims)
    for i in range(n):
        left = max(0, i - window_size // 2)
        right = min(n, i + window_size // 2 + 1)
        smoothed[i] = float(np.mean(max_sims[left:right]))

    supported = np.sum(smoothed >= threshold)
    result["score"] = float(supported / len(sentences))
    result["sentences"] = sentences
    result["sentence_scores"] = [round(float(s), 4) for s in smoothed]

    # 找最低分句子
    min_idx = int(np.argmin(smoothed))
    result["lowest_sentence"] = sentences[min_idx]
    result["lowest_score"] = round(float(smoothed[min_idx]), 4)

    return result


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


async def run_ragas_evaluation(records: list[dict]) -> dict:
    """
    混合评估策略：
      - ContextPrecision: query vs context 的 embedding 余弦相似度
      - ContextRecall: ground_truth 逐句在 contexts 中的覆盖率（替代原二元判定）
      - Faithfulness: LLM Judge（gpt-4o）逐句判断，失败降级到 embedding 余弦相似度
      - AnswerRelevancy: 答案与问题的语义相似度
      - 检索指标 Recall@K / Precision@K
      - InfoPoint 覆盖率: ground_truth 关键信息点在回答中的覆盖比例
    """
    # ── 1. ContextPrecision（不变） & ContextRecall（升级版）──
    logger.info("[Eval] 阶段1/5: 计算 ContextPrecision & ContextRecall...")
    ctx_precision = []
    ctx_recall = []
    for i, r in enumerate(records):
        if i > 0 and i % 10 == 0:
            logger.info(f"  ContextPrecision/Recall: {i}/{len(records)}")
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

        c_arr = np.array(c_embs)
        c_norm = c_arr / (np.linalg.norm(c_arr, axis=1, keepdims=True) + 1e-12)

        sims = [_cosine_sim(q_emb, ce) for ce in c_embs]
        # precision: 相关 context 的比例
        relevant = sum(1 for s in sims if s >= EVAL_THRESHOLD)
        ctx_precision.append(round(relevant / len(sims), 4))

        # recall（升级版）：用 ground_truth 逐句判定覆盖率，替代"至少有一个相关"的二元逻辑
        gt = r.get("ground_truth", "")
        gt_sents = [s.strip() for s in re.split(r'[。！？\n;；]', gt) if len(s.strip()) > 2]
        if gt_sents:
            gt_embs = EmbeddingService.embed_texts(gt_sents)
            gt_arr = np.array(gt_embs)
            gt_norm = gt_arr / (np.linalg.norm(gt_arr, axis=1, keepdims=True) + 1e-12)
            sim_gt = np.dot(gt_norm, c_norm.T)  # (n_gt, n_contexts)
            covered = np.max(sim_gt, axis=1) >= EVAL_THRESHOLD
            ctx_recall.append(round(float(np.mean(covered)), 4))
        else:
            ctx_recall.append(round(1.0 if relevant > 0 else 0.0, 4))

    # ── 2. Faithfulness + 句子级详情 + 多阈值敏感性 ──
    logger.info("[Eval] 阶段2/5: Faithfulness LLM Judge（并发 5 路，单次超时 60s）...")
    async def _eval_one_faithfulness(r: dict) -> tuple:
        f_result = await _compute_faithfulness(r["answer"], r.get("contexts", []), question=r["question"])
        return round(f_result["score"], 4), f_result

    sem_faith = asyncio.Semaphore(EVAL_CONCURRENCY)
    async def _limited_faith(r: dict) -> tuple:
        async with sem_faith:
            return await _eval_one_faithfulness(r)

    faithful_results = await asyncio.gather(*[_limited_faith(r) for r in records])
    custom_faithfulness = [fr[0] for fr in faithful_results]
    faithful_details = [fr[1] for fr in faithful_results]
    success_count = sum(1 for f in faithful_details if f.get("sentences"))
    logger.info(f"  Faithfulness 完成: {success_count}/{len(records)} 条使用 LLM Judge, 其余降级到 embedding")

    # 多阈值敏感性分析（可选，耗时较长，默认跳过；需要时取消注释）
    # thresholds_for_analysis = [0.35, 0.38, 0.45]
    # multi_threshold_scores = {}
    # for t in thresholds_for_analysis:
    #     scores = []
    #     for r in records:
    #         scores.append(_compute_faithfulness_embedding(r["answer"], r.get("contexts", []), threshold=t)["score"])
    #     multi_threshold_scores[str(t)] = round(float(np.mean(scores)), 4)
    multi_threshold_scores = {}

    # ── 3. AnswerRelevancy（本地 embedding，串行即可）──
    custom_relevancy = []
    for i, r in enumerate(records):
        if i > 0 and i % 10 == 0:
            logger.info(f"  AnswerRelevancy: {i}/{len(records)}")
        custom_relevancy.append(
            round(_compute_answer_relevancy(r["question"], r["answer"]), 4)
        )

    # ── 4. 检索指标 ──
    logger.info("[Eval] 阶段4/5: 检索指标 (Recall@K / Precision@K)...")
    retrieval_metrics = _compute_retrieval_metrics(records)

    # ── 5. InfoPoint 覆盖率（P2 优化）──
    logger.info("[Eval] 阶段5/5: InfoPoint 覆盖率...")
    info_point_coverages = []
    for i, r in enumerate(records):
        if i > 0 and i % 10 == 0:
            logger.info(f"  InfoPoint: {i}/{len(records)}")
        gt = r.get("ground_truth", "")
        answer = r.get("answer", "")
        # 自动从 ground_truth 提取关键信息点（按标点分句）
        info_points = [s.strip() for s in re.split(r'[。！？\n;；]', gt) if len(s.strip()) > 3]
        if not info_points:
            info_point_coverages.append(1.0)
            continue
        if not answer:
            info_point_coverages.append(0.0)
            continue

        ip_embs = EmbeddingService.embed_texts(info_points)
        a_emb = EmbeddingService.embed_query(answer)
        ip_arr = np.array(ip_embs)
        ip_norm = ip_arr / (np.linalg.norm(ip_arr, axis=1, keepdims=True) + 1e-12)
        a_norm = np.array(a_emb) / (np.linalg.norm(np.array(a_emb)) + 1e-12)
        sims_ip = np.dot(ip_norm, a_norm)  # (n_points,)
        covered_ip = np.sum(sims_ip >= EVAL_THRESHOLD)
        info_point_coverages.append(round(float(covered_ip / len(info_points)), 4))

    # ── 6. 合并结果 ──
    logger.info("[Eval] 阶段3-5/5: 完成 AnswerRelevancy / 检索指标 / InfoPoint覆盖率")
    summary = {
        "context_precision": round(float(np.mean(ctx_precision)), 4),
        "context_recall": round(float(np.mean(ctx_recall)), 4),
        "faithfulness": round(float(np.mean(custom_faithfulness)), 4),
        "answer_relevancy": round(float(np.mean(custom_relevancy)), 4),
        "retrieval/recall@k": retrieval_metrics["recall_at_k"],
        "retrieval/precision@k": retrieval_metrics["precision_at_k"],
        "info_point_coverage": round(float(np.mean(info_point_coverages)), 4),
        "faithfulness_threshold_sweep": multi_threshold_scores,
    }

    # 难例集（P2 优化）：挑出跨章节综合 + 边界条件 + 口语化的记录
    hard_categories = {"跨章节综合查询", "边界条件查询", "口语化查询"}

    detail_rows = []
    hard_rows = []
    for i, r in enumerate(records):
        row = {
            "id": r["id"],
            "category": r["category"],
            "question": r["question"][:40],
            "context_precision": ctx_precision[i] if i < len(ctx_precision) else 0.0,
            "context_recall": ctx_recall[i] if i < len(ctx_recall) else 0.0,
            "faithfulness": custom_faithfulness[i] if i < len(custom_faithfulness) else 0.0,
            "answer_relevancy": custom_relevancy[i] if i < len(custom_relevancy) else 0.0,
            "info_point_coverage": info_point_coverages[i] if i < len(info_point_coverages) else 0.0,
            "faithful_lowest_sentence": faithful_details[i].get("lowest_sentence", "")[:60] if i < len(faithful_details) else "",
            "faithful_lowest_score": faithful_details[i].get("lowest_score", 1.0) if i < len(faithful_details) else 1.0,
        }
        detail_rows.append(row)
        if r["category"] in hard_categories:
            hard_rows.append(row)

    return {
        "summary": summary,
        "detail": detail_rows,
        "hard_detail": hard_rows,
    }


# ──────────────────────────────────────────────────────
# 报告输出
# ──────────────────────────────────────────────────────


def print_report(summary: dict, detail: list, records: list, rerank_enabled: bool, eval_duration_ms: float = 0):
    """终端打印评估报告"""
    total = len(records)
    avg_retrieval = sum(r.get("retrieval_ms", 0) for r in records) / max(total, 1)
    avg_total = sum(r.get("total_ms", 0) for r in records) / max(total, 1)

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  RAG 系统评估报告 — 员工考勤管理制度")
    print(f"  测试集条数: {total} | 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Reranker: {settings.RERANK_MODEL} {'(启用)' if rerank_enabled else '(关闭)'}")
    print(f"  生成LLM: {settings.DEFAULT_LLM_PROVIDER}")
    print(f"{sep}")

    # 主指标
    print(f"  ── [Embedding] 语义相似度指标──")
    for k in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "info_point_coverage"]:
        if k in summary:
            print(f"  {k:<24s}: {summary[k]:.4f}")

    print(f"  ── [Retrieval] 检索质量 ──")
    for k in ["retrieval/recall@k", "retrieval/precision@k"]:
        if k in summary:
            print(f"  {k:<24s}: {summary[k]:.4f}")

    # 多阈值敏感性分析
    if "faithfulness_threshold_sweep" in summary and summary["faithfulness_threshold_sweep"]:
        sweep = summary["faithfulness_threshold_sweep"]
        print(f"  ── [Faithfulness] 多阈值敏感性 (Embedding) ──")
        for t_str in sorted(sweep.keys()):
            print(f"  threshold={t_str:<4s} → {sweep[t_str]:.4f}")
        vals = [sweep[k] for k in sorted(sweep.keys())]
        spread = max(vals) - min(vals)
        print(f"  极差: {spread:.4f} {'✅ 稳定' if spread < 0.05 else '⚠️ 注意: 阈值敏感'}")
        print(f"  {'─' * 60}")

    print(f"  平均检索耗时             : {avg_retrieval:.1f}ms")
    print(f"  平均端到端耗时           : {avg_total:.1f}ms")
    print(f"  评估计算耗时             : {eval_duration_ms:.0f}ms")
    print(f"{sep}")

    # 按分类汇总
    cat_scores = {}
    for row in detail:
        cat = row["category"]
        cat_scores.setdefault(cat, {"count": 0, "precision": 0, "recall": 0, "faithfulness": 0, "relevancy": 0, "ip": 0})
        d = cat_scores[cat]
        d["count"] += 1
        d["precision"] += row["context_precision"]
        d["recall"] += row["context_recall"]
        d["faithfulness"] += row["faithfulness"]
        d["relevancy"] += row["answer_relevancy"]
        d["ip"] += row.get("info_point_coverage", 0)

    print(f"\n  [Category] 按类别分析:")
    print(f"  {'类别':<16s} {'条数':>4s}  {'Precision':>10s}  {'Recall':>8s}  {'Faithful':>9s}  {'Relevancy':>9s}  {'IP_Cov':>7s}")
    print(f"  {'─' * 70}")
    for cat, d in cat_scores.items():
        n = d["count"]
        print(f"  {cat:<16s} {n:>4d}  {d['precision']/n:>10.4f}  {d['recall']/n:>8.4f}  {d['faithfulness']/n:>9.4f}  {d['relevancy']/n:>9.4f}  {d['ip']/n:>7.4f}")

    # 低分句子告警
    low_sentences = [row for row in detail if row.get("faithful_lowest_score", 1.0) < 0.45 and row["faithfulness"] < 1.0]
    if low_sentences:
        print(f"\n  [WARNING] Faithfulness 低分句子告警:")
        for row in low_sentences[:5]:
            print(f"    id={row['id']:>2d} {row['question']:<20s} score={row['faithful_lowest_score']:.3f}")
            print(f"    最低分句: {row['faithful_lowest_sentence'][:50]}")
            print()


def save_markdown(summary: dict, detail: list, records: list, rerank_enabled: bool, hard_detail: list = None, eval_duration_ms: float = 0, report_path: Path = None):
    """保存 Markdown 评估报告"""
    if report_path is None:
        report_path = REPORT_BASE_DIR / "V3-三级流水线(全量)" / "V3_eval_report.md"
    total = len(records)
    avg_retrieval = sum(r.get("retrieval_ms", 0) for r in records) / max(total, 1)
    avg_total = sum(r.get("total_ms", 0) for r in records) / max(total, 1)

    lines = []
    lines.append("# RAG 系统评估报告\n")
    lines.append(f"> 测试集: {total} 条 | 时间: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Reranker: `{settings.RERANK_MODEL}` {'[OK] 启用' if rerank_enabled else '[ERROR] 关闭'}")
    lines.append(f"> LLM: `{settings.DEFAULT_LLM_PROVIDER}` | Faithfulness: `LLM Judge (deepseek-chat)` | 其余指标: `Embedding 余弦相似度`\n")

    lines.append("## 综合指标\n")
    lines.append("### 语义相似度指标\n")
    lines.append("| 指标 | 得分 |")
    lines.append("|------|------|")
    for k in ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "info_point_coverage"]:
        if k in summary:
            lines.append(f"| {k} | {summary[k]:.4f} |")

    lines.append("\n### 检索质量\n")
    lines.append("| 指标 | 得分 |")
    lines.append("|------|------|")
    for k in ["retrieval/recall@k", "retrieval/precision@k"]:
        if k in summary:
            lines.append(f"| {k} | {summary[k]:.4f} |")

    # 多阈值敏感性
    if "faithfulness_threshold_sweep" in summary and summary["faithfulness_threshold_sweep"]:
        sweep = summary["faithfulness_threshold_sweep"]
        lines.append("\n### Faithfulness 多阈值敏感性分析（Embedding 降级模式）\n")
        lines.append("| 阈值 | faithfulness |")
        lines.append("|------|-------------|")
        for t_str in sorted(sweep.keys()):
            lines.append(f"| {t_str} | {sweep[t_str]:.4f} |")
        vals = [sweep[k] for k in sorted(sweep.keys())]
        spread = max(vals) - min(vals)
        lines.append(f"\n> 极差: {spread:.4f} — {'✅ 阈值稳定' if spread < 0.05 else '⚠️ 阈值敏感，建议关注'}")

    lines.append(f"\n### 性能\n")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 平均检索耗时 | {avg_retrieval:.1f}ms |")
    lines.append(f"| 平均端到端耗时 | {avg_total:.1f}ms |")
    lines.append(f"| 评估计算耗时 | {eval_duration_ms:.0f}ms |\n")

    # 分类汇总
    cat_scores = {}
    for row in detail:
        cat = row["category"]
        cat_scores.setdefault(cat, {"count": 0, "precision": 0, "recall": 0, "faithfulness": 0, "relevancy": 0, "ip": 0})
        d = cat_scores[cat]
        d["count"] += 1
        d["precision"] += row["context_precision"]
        d["recall"] += row["context_recall"]
        d["faithfulness"] += row["faithfulness"]
        d["relevancy"] += row["answer_relevancy"]
        d["ip"] += row.get("info_point_coverage", 0)

    lines.append("## [Category] 按类别分析\n")
    lines.append("| 类别 | 条数 | context_precision | context_recall | faithfulness | answer_relevancy | info_point_coverage |")
    lines.append("|------|------|------------------|---------------|--------------|------------------|---------------------|")
    for cat, d in cat_scores.items():
        n = d["count"]
        lines.append(f"| {cat} | {n} | {d['precision']/n:.4f} | {d['recall']/n:.4f} | {d['faithfulness']/n:.4f} | {d['relevancy']/n:.4f} | {d['ip']/n:.4f} |")

    # 难例集
    if hard_detail:
        lines.append("\n## [Hard] 难例集监控\n")
        lines.append("| # | 类别 | 问题 | c_precision | c_recall | faithful | relevancy | IP_coverage |")
        lines.append("|---|------|------|------------|---------|----------|----------|-------------|")
        for row in hard_detail:
            lines.append(
                f"| {row['id']} | {row['category']} | {row['question']} "
                f"| {row['context_precision']:.4f} | {row['context_recall']:.4f} "
                f"| {row['faithfulness']:.4f} | {row['answer_relevancy']:.4f} "
                f"| {row.get('info_point_coverage', 0):.4f} |"
            )

    lines.append("\n## [Detail] 逐条详情\n")
    lines.append("| # | 类别 | 问题 | c_precision | c_recall | faithful | relevancy | IP_cov | lowest_sent_score | lowest_sentence |")
    lines.append("|---|------|------|------------|---------|----------|----------|--------|-------------------|-----------------|")
    for row in detail:
        lines.append(
            f"| {row['id']} | {row['category']} | {row['question']} "
            f"| {row['context_precision']:.4f} | {row['context_recall']:.4f} "
            f"| {row['faithfulness']:.4f} | {row['answer_relevancy']:.4f} "
            f"| {row.get('info_point_coverage', 0):.4f} "
            f"| {row.get('faithful_lowest_score', 1.0):.4f} "
            f"| {row.get('faithful_lowest_sentence', '')[:40]} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [Report] 报告已保存: {report_path}")


def save_json_report(summary: dict, detail: list, records: list, rerank_enabled: bool, report_json_path: Path = None):
    """保存 JSON 格式评估结果，用于自动对比"""
    if report_json_path is None:
        report_json_path = REPORT_BASE_DIR / "V3-三级流水线(全量)" / "V3_eval_report.json"
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
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [Report] JSON 报告已保存: {report_json_path}")


def save_baseline(report_json_path: Path, baseline_path: Path):
    """将当前 JSON 报告另存为基线"""
    import shutil
    if report_json_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(report_json_path, baseline_path)
        print(f"  [Baseline] 基线已保存: {baseline_path}")
    else:
        print(f"  [ERROR] 未找到 JSON 报告，请先运行评估")


def auto_compare_with_baseline(summary: dict, baseline_path: Path):
    """自动与基线对比并打印差分"""
    if not baseline_path.exists():
        print(f"\n  [Baseline] 无基线文件，跳过对比。运行 `python {__file__} --save-baseline` 保存基线。")
        return

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
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


MAX_CONCURRENT = 3
EVAL_TOP_K = 8
EMBED_BATCH_SIZE = 5
EVAL_CONCURRENCY = 5  # 评估阶段 Faithfulness LLM Judge 并发数


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


async def main(version: str = "V3"):
    """运行指定版本的评估"""
    # ── 获取版本预设并覆盖配置 ──
    preset = VERSION_PRESETS.get(version)
    if preset is None:
        print(f"[ERROR] 未知版本: {version}，可选: {list(VERSION_PRESETS.keys())}")
        return

    # 覆盖 settings（运行时修改，不影响 .env 文件）
    settings.RERANK_ENABLED = preset["rerank_enabled"]
    settings.VECTOR_WEIGHT = preset["vector_weight"]
    settings.BM25_WEIGHT = preset["bm25_weight"]
    settings.RETRIEVAL_SCORE_THRESHOLD = preset["score_threshold"]
    settings.RERANK_MULTIPLIER = preset["rerank_multiplier"]

    # ── 计算输出路径 ──
    version_name = preset["name"]
    report_dir = REPORT_BASE_DIR / version_name
    report_path = report_dir / f"{version}_eval_report.md"
    report_json_path = report_dir / f"{version}_eval_report.json"
    baseline_path = report_dir / f"{version}_baseline.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  版本: {version} — {version_name}")
    print(f"  配置: rerank={preset['rerank_enabled']}, "
          f"vec_weight={preset['vector_weight']}, "
          f"bm25_weight={preset['bm25_weight']}, "
          f"threshold={preset['score_threshold']}, "
          f"rerank_mult={preset['rerank_multiplier']}")
    print(f"  输出: {report_dir}")
    print(f"{'='*60}")

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
    t_eval_start = time.perf_counter()
    eval_result = await run_ragas_evaluation(records)  # 所有记录都参与评估
    eval_duration_ms = (time.perf_counter() - t_eval_start) * 1000

    rerank_enabled = getattr(settings, "RERANK_ENABLED", False)
    print_report(eval_result["summary"], eval_result["detail"], records, rerank_enabled, eval_duration_ms)
    save_markdown(eval_result["summary"], eval_result["detail"], records, rerank_enabled, eval_result.get("hard_detail"), eval_duration_ms, report_path)
    save_json_report(eval_result["summary"], eval_result["detail"], records, rerank_enabled, report_json_path)

    # 与基线自动对比
    auto_compare_with_baseline(eval_result["summary"], baseline_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 系统评估脚本 — 支持 A/B 版本对比")
    parser.add_argument("--version", type=str, default="V3",
                        choices=list(VERSION_PRESETS.keys()),
                        help=f"评估版本预设: {list(VERSION_PRESETS.keys())}")
    parser.add_argument("--save-baseline", action="store_true",
                        help="将评估结果另存为基线")

    args = parser.parse_args()

    # 运行评估
    asyncio.run(main(version=args.version))

    # 保存基线（如果需要）
    if args.save_baseline:
        preset = VERSION_PRESETS[args.version]
        version_name = preset["name"]
        report_json = REPORT_BASE_DIR / version_name / f"{args.version}_eval_report.json"
        baseline_path = REPORT_BASE_DIR / version_name / f"{args.version}_baseline.json"
        save_baseline(report_json, baseline_path)
