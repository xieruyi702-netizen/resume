"""上下文压缩：检索块按分数贪心装入预算 + 失败记忆截断。

compress_on=False 时原样返回，保证 off/on 消融可对比。
"""
from __future__ import annotations

from src.context.budget import BUDGET, chars


def clip_memory_hits(hits: list[dict], budget: int | None = None) -> list[dict]:
    """失败记忆注入：按相似度降序，装不下就丢掉最低分。"""
    if not hits:
        return []
    budget = BUDGET["failure_fewshot"] if budget is None else budget
    ordered = sorted(hits, key=lambda h: -float(h.get("similarity") or 0))
    kept, used = [], 0
    for h in ordered:
        line = (
            f"- 相似问题「{h.get('question', '')}」曾失败({h.get('outcome', '')})，"
            f"根因: {h.get('root_cause', '')}；教训: {h.get('lesson', '')}"
        )
        n = chars(line) + 1
        if kept and used + n > budget:
            break
        if not kept and n > budget:  # 至少留 1 条，硬截断
            h = {**h, "lesson": (h.get("lesson") or "")[: max(40, budget // 4)]}
        kept.append(h)
        used += n
    return kept


def pack_chunks(chunks: list, budget: int | None = None) -> tuple[list, dict]:
    """按 score 降序、parent 去重后贪心装入预算。

    返回 (kept_chunks, stats)。
    """
    budget = BUDGET["retrieval"] if budget is None else budget
    if not chunks:
        return [], {"before_n": 0, "after_n": 0, "before_chars": 0, "after_chars": 0, "dropped": 0}

    # parent 去重：保留最高分
    best: dict[str, object] = {}
    for c in chunks:
        pid = getattr(c, "parent_id", None) or id(c)
        prev = best.get(pid)
        if prev is None or float(getattr(c, "score", 0)) > float(getattr(prev, "score", 0)):
            best[pid] = c
    ranked = sorted(best.values(), key=lambda c: -float(getattr(c, "score", 0)))

    before_chars = sum(chars(getattr(c, "parent_text", "")) for c in ranked)
    kept, used = [], 0
    for c in ranked:
        # 格式化开销粗算：前缀约 30 字
        n = chars(getattr(c, "parent_text", "")) + 30
        if kept and used + n > budget:
            continue
        if not kept and n > budget:
            # 首块超预算：截断 parent_text（不改原对象）
            text = getattr(c, "parent_text", "") or ""
            cut = max(80, budget - 30)
            from copy import copy
            c = copy(c)
            c.parent_text = text[:cut] + "…"
            n = chars(c.parent_text) + 30
        kept.append(c)
        used += n

    after_chars = sum(chars(getattr(c, "parent_text", "")) for c in kept)
    stats = {
        "before_n": len(ranked),
        "after_n": len(kept),
        "before_chars": before_chars,
        "after_chars": after_chars,
        "dropped": max(0, len(ranked) - len(kept)),
        "budget": budget,
    }
    return kept, stats


def format_evidence(chunks: list) -> str:
    if not chunks:
        return "(检索无结果)"
    parts = [f"[{i}] (来源: {c.doc_id}) {c.parent_text}" for i, c in enumerate(chunks, 1)]
    return "\n\n".join(parts)
