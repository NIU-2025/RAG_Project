"""
多轮对话 Query 改写：按需触发，三层过滤。

- 第一层：无历史对话 → 跳过
- 第二层：规则检测（零成本）→ 判断是否需要改写
- 第三层：LLM 改写（仅 ~15-20% 的追问触发）
"""

import re
import time
from typing import List

from app.core.logger import logger


REWRITE_SYSTEM_PROMPT = """基于以下对话历史，将用户的追问改写为一个独立的、不需要上下文就能理解的完整问题。只输出改写后的问题，不要任何解释。"""


# ─── 第二层：规则检测 ───

_REF_PATTERNS = [
    re.compile(r"^(那|它|这|这些|那些|那个|这个|还有)\s*"),
    re.compile(r"^(怎么|如何|什么|多少|谁|哪|为什)"),
]

_SHORT_FOLLOWUP_MAX = 8
_COMPLETE_SENTENCE_MIN = 10


def _needs_rewrite(query: str, history: List) -> bool:
    """基于规则的快速判断。返回 True 则需要 LLM 改写。"""
    q = query.strip()

    # 规则 1：含明确指代词
    if any(w in q for w in ("它", "这个", "那个", "这些", "那些", "上面说的", "前面的")):
        return True

    # 规则 2：以"那"/"这"/"还有"开头且字数少
    if re.match(r"^(那|这|还有)\s*", q) and len(q) <= 8:
        return True

    # 规则 3：极短追问（≤5 字且含疑问词）
    if len(q) <= 5 and any(w in q for w in ("怎么", "什么", "多少", "谁", "哪", "嘛", "呢", "吗", "不")):
        return True

    # 规则 4：明显缺少主语的省略句（问句且 <8 字但无具体名词）
    if len(q) <= _SHORT_FOLLOWUP_MAX and q.endswith(("?", "？", "呢", "吗", "嘛")):
        has_specific_noun = re.search(r"[^\u4e00-\u9fa5]|[的是在有了]", q)
        if not has_specific_noun or _has_only_pronouns(q):
            return True

    # 默认：完整句子，不需要改写
    return False


def _has_only_pronouns(text: str) -> bool:
    """检查是否只含代词/虚词而无实质名词"""
    stop_words = {"它", "这", "那", "什么", "怎么", "为什么", "多少", "谁", "哪", "吗", "呢", "了", "的", "啊"}
    meaningful = [c for c in text if c not in stop_words and c not in "，。！？""''：；（《》）"]
    return len("".join(meaningful)) < 3


# ─── 第三层：LLM 改写 ───


async def _llm_rewrite(query: str, provider: str, history_messages: List[dict], db=None) -> str:
    """调用 LLM 将追问改写成独立问题"""
    from app.services.llm import LLMService

    rew_messages = [{"role": "system", "content": REWRITE_SYSTEM_PROMPT}]

    for msg in history_messages[-5:]:
        rew_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        t0 = time.perf_counter()
        rewritten, _ = await LLMService.chat(
            provider=provider,
            model=None,
            messages=rew_messages,
            user_message=query,
            context="",
            db=db,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        rewritten = rewritten.strip()
        if rewritten and rewritten != query.strip():
            logger.info(f"Query改写: [{query[:30]}] → [{rewritten[:50]}] ({elapsed:.0f}ms)")
            return rewritten
        else:
            logger.debug(f"Query改写: LLM 未产生有效改写，保留原query")
            return query
    except Exception as e:
        logger.warning(f"Query改写失败，降级使用原 query: {e}")
        return query


# ─── 主入口 ───


async def rewrite_query_if_needed(
    current_query: str,
    history: List,
    provider: str,
    db=None,
) -> str:
    """
    按需改写多轮对话中的用户追问。

    返回:
        改写后的 query（如果不需要改写则返回原 query）
    """
    query = current_query.strip()

    # ── 第一层：无历史对话 → 跳过 ──
    if not history or len(history) <= 1:
        logger.debug("Query改写: 跳过（无历史对话）")
        return query

    # ── 第二层：规则检测 → 快速判断 ──
    if not _needs_rewrite(query, history):
        logger.debug("Query改写: 跳过（规则判定无需改写）")
        return query

    # ── 第三层：LLM 改写 ──
    logger.debug(f"Query改写: 触发LLM改写 [{query[:30]}]")
    return await _llm_rewrite(query, provider, history, db=db)
