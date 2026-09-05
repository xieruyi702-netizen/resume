"""共享编码器：bge-small-zh 单例。

Chroma 自带的默认编码器是英文 MiniLM（384 维），与库内 bge 向量（512 维）
不匹配、对中文也不友好——所有 add/query 一律走这里显式编码。
"""
from sentence_transformers import SentenceTransformer

from src import config

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()
