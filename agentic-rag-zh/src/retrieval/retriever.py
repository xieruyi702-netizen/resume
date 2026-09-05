"""混合检索：dense(Chroma) + BM25 → RRF 融合 → 可选 BGE 重排 → 返回 parent。

child 用于打分排序，parent（完整上下文）返回给 LLM——这是父子分块的意义。
"""
import json
import pickle

import chromadb
import jieba

from src import config


class RetrievedChunk:
    def __init__(self, doc_id: str, parent_id: str, parent_text: str, child_text: str, score: float):
        self.doc_id = doc_id
        self.parent_id = parent_id
        self.parent_text = parent_text
        self.child_text = child_text
        self.score = round(score, 6)

    def __repr__(self):
        return f"<{self.doc_id} score={self.score}>"


class Retriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.col = client.get_or_create_collection(
            config.KB_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        with open(config.BM25_PATH, "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        self.id2parent = dict(zip(blob["child_ids"], blob["child_parents"]))
        with open(config.PARENTS_PATH, encoding="utf-8") as f:
            self.parents = json.load(f)
        self._child_cache: dict[str, str] = {}

    def child_text(self, child_id: str) -> str:
        if child_id not in self._child_cache:
            self._child_cache[child_id] = self.col.get(ids=[child_id])["documents"][0]
        return self._child_cache[child_id]

    def retrieve(self, query: str, k: int = config.TOP_K,
                 candidate_k: int = config.CANDIDATE_K, use_rerank: bool | None = None) -> list[RetrievedChunk]:
        from src.retrieval.embedder import encode
        n = min(candidate_k, self.col.count())
        dense_ids = self.col.query(query_embeddings=encode([query]), n_results=n)["ids"][0]
        bm_ids = self.bm25.get_top_n(jieba.lcut(query), list(self.id2parent.keys()), n=n)

        rrf: dict[str, float] = {}
        for r, cid in enumerate(dense_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (config.RRF_K + r + 1)
        for r, cid in enumerate(bm_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (config.RRF_K + r + 1)
        ranked = sorted(rrf.items(), key=lambda x: -x[1])

        if use_rerank is None:
            use_rerank = config.USE_RERANK
        if use_rerank:
            from src.retrieval.reranker import rerank
            ranked = rerank(query, ranked, self)

        # 同一 parent 的多个 child 只保留最高分的一个，top-k 不浪费槽位
        seen_parents: set[str] = set()
        deduped = []
        for cid, sc in ranked:
            pid = self.id2parent[cid]
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            deduped.append((cid, sc))
            if len(deduped) >= k:
                break

        out = []
        for cid, sc in deduped:
            pid = self.id2parent[cid]
            doc_id = pid.split("#")[0]
            out.append(RetrievedChunk(doc_id, pid, self.parents[pid], self.child_text(cid), sc))
        return out


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
