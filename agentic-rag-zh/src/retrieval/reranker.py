"""BGE cross-encoder 重排（可选，USE_RERANK=1 启用）。

模型 ~1.1GB 首次启用时自动下载；对 RRF 融合后的候选按 (query, child) 精排。
"""
from src import config

_cross = None


def rerank(query: str, ranked: list[tuple[str, float]], retriever) -> list[tuple[str, float]]:
    global _cross
    if _cross is None:
        from sentence_transformers import CrossEncoder
        print(f"[rerank] 加载 {config.RERANK_MODEL} ...")
        _cross = CrossEncoder(config.RERANK_MODEL, max_length=512)
    pairs, items = [], []
    for cid, _ in ranked:
        pairs.append((query, retriever.child_text(cid)))
        items.append(cid)
    scores = _cross.predict(pairs)
    order = sorted(range(len(items)), key=lambda i: -scores[i])
    return [(items[i], float(scores[i])) for i in order]
