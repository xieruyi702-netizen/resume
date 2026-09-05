"""评测指标：检索层（Hit@k / MRR / Recall@k / nDCG@k）+ 答案层（关键词/拒答）。

单相关文档时 Recall@k 退化为 Hit@k；多跳 case（多个 expected_sources）上
Recall@k / nDCG@k 才有区分度——这是相对参考实现新增的两个指标。
"""
import math

REFUSAL_PHRASES = (
    "没有找到", "未找到", "没有相关", "无法回答", "知识库中没有", "知识库中没",
    "无法从知识库", "超出了知识库", "超出知识库", "无法确定", "没有足够",
)


def hit_at_k(sources: list[str], expected: list[str], k: int) -> int:
    if not expected:
        return 0
    top = set(sources[:k])
    return 1 if any(s in top for s in expected) else 0


def reciprocal_rank(sources: list[str], expected: list[str]) -> float:
    for i, s in enumerate(sources[: len(expected) * 3 + 5], start=1):
        if s in expected:
            return 1.0 / i
    return 0.0


def recall_at_k(sources: list[str], expected: list[str], k: int) -> float:
    """多相关文档：top-k 命中的相关文档数 / 相关文档总数。"""
    if not expected:
        return 0.0
    top = set(sources[:k])
    return sum(1 for s in expected if s in top) / len(expected)


def ndcg_at_k(sources: list[str], expected: list[str], k: int) -> float:
    """排名质量：命中越靠前贡献越高。"""
    if not expected:
        return 0.0
    exp = set(expected)
    dcg = sum(1.0 / math.log2(i + 2) for i, s in enumerate(sources[:k]) if s in exp)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(exp), k)))
    return dcg / idcg if idcg else 0.0


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    a = (answer or "").lower()
    return all(kw.lower() in a for kw in keywords)


def refusal_ok(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p in a for p in REFUSAL_PHRASES)
