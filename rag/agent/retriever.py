#!/usr/bin/env python3
"""美食汇双路检索：BM25（jieba）+ 向量（智谱 embedding-3）→ RRF 融合。"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None


def tokenize(text: str) -> list[str]:
    text = text.lower()
    if jieba is not None:
        return [t.strip() for t in jieba.lcut_for_search(text) if t.strip() and not t.isspace()]
    parts = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    return [p for p in parts if p]


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_len = [len(t) for t in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.df: Counter[str] = Counter()
        for toks in self.docs_tokens:
            for t in set(toks):
                self.df[t] += 1
        self.N = len(chunks)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = []
        for i, toks in enumerate(self.docs_tokens):
            tf = Counter(toks)
            score = 0.0
            dl = self.doc_len[i] or 1
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                idf = self._idf(term)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[i], s) for i, s in scores[:top_k]]


class VectorIndex:
    """预计算 L2 归一化向量 + 在线 embedding 查询，余弦 = 点积。"""

    def __init__(self, chunks: list[dict], matrix: np.ndarray, chunk_ids: list[str], embedder=None):
        if len(chunks) != matrix.shape[0]:
            # 允许按 chunk_id 对齐
            by_id = {c["chunk_id"]: c for c in chunks}
            aligned = []
            keep = []
            for i, cid in enumerate(chunk_ids):
                if cid in by_id:
                    aligned.append(by_id[cid])
                    keep.append(i)
            chunks = aligned
            matrix = matrix[keep]
            chunk_ids = [chunk_ids[i] for i in keep]
        self.chunks = chunks
        self.matrix = matrix.astype(np.float32)
        self.chunk_ids = chunk_ids
        self.embedder = embedder
        self._id_to_pos = {cid: i for i, cid in enumerate(chunk_ids)}

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if self.embedder is None:
            return []
        qv = np.asarray(self.embedder.embed(query)[0], dtype=np.float32)
        qv = qv / max(float(np.linalg.norm(qv)), 1e-12)
        sims = self.matrix @ qv
        if top_k >= len(sims):
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, top_k)[:top_k]
            idx = idx[np.argsort(-sims[idx])]
        out = []
        for i in idx[:top_k]:
            out.append((self.chunks[int(i)], float(sims[int(i)])))
        return out


def rrf_fuse(
    ranked_lists: list[list[tuple[dict, float]]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion。"""
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            best[cid] = chunk
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(best[cid], sc) for cid, sc in ordered[:top_k]]


class HybridIndex:
    """双路：BM25 + Dense，RRF 融合；向量索引缺失时退化为纯 BM25。"""

    def __init__(self, bm25: BM25Index, vector: VectorIndex | None = None, rrf_k: int = 60):
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.chunks = bm25.chunks

    @property
    def modes(self) -> list[str]:
        modes = ["bm25"]
        if self.vector is not None:
            modes.append("dense")
        return modes

    def search(self, query: str, top_k: int = 5, mode: str | None = None) -> list[tuple[dict, float]]:
        mode = mode or os.environ.get("MEISHI_RETRIEVAL", "hybrid")
        pool = max(top_k * 4, 20)
        if mode == "bm25" or self.vector is None:
            return self.bm25.search(query, top_k=top_k)
        if mode == "dense":
            return self.vector.search(query, top_k=top_k)
        # hybrid
        a = self.bm25.search(query, top_k=pool)
        b = self.vector.search(query, top_k=pool)
        return rrf_fuse([a, b], k=self.rrf_k, top_k=top_k)


def load_chunks(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def default_chunks_path() -> Path:
    return Path(__file__).resolve().parents[1] / "kb" / "chunks.jsonl"


def default_embed_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1] / "kb"
    return root / "embeddings.npy", root / "embeddings_meta.json"


def build_index(chunks_path: Path | None = None, use_vector: bool = True) -> HybridIndex:
    path = chunks_path or default_chunks_path()
    chunks = load_chunks(path)
    bm25 = BM25Index(chunks)
    vector = None
    if use_vector:
        npy, meta_p = default_embed_paths()
        if npy.exists() and meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            mat = np.load(npy)
            try:
                from embedder import get_embedder
                embedder = get_embedder(meta.get("backend") or "local")
            except Exception:
                embedder = None
            if embedder is not None:
                vector = VectorIndex(chunks, mat, meta.get("chunk_ids") or [c["chunk_id"] for c in chunks], embedder)
    return HybridIndex(bm25, vector)
