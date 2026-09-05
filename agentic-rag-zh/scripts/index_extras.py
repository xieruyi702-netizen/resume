"""增量同步索引：把 kb_docs.jsonl 中尚未入库的文档补进 Chroma，并重建 BM25。

用法（kb_docs.jsonl 追加了新文档后跑一次）：
  uv run python scripts/index_extras.py

BM25 不支持增量插入，故全量重建（重分词约 5-8 分钟）；Chroma 只编码新增部分。
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
import jieba
from rank_bm25 import BM25Okapi

from src import config
from src.ingest.chunker import split_doc
from src.retrieval.embedder import get_embedder


def main():
    docs = [json.loads(line) for line in open(config.CORPUS_PATH, encoding="utf-8")]
    parents = json.load(open(config.PARENTS_PATH, encoding="utf-8"))
    indexed = {pid.split("#")[0] for pid in parents}
    missing = [d for d in docs if d["doc_id"] not in indexed]
    print(f"[sync] 库内 {len(indexed)} 篇，语料 {len(docs)} 篇，待补 {len(missing)} 篇")
    if not missing:
        print("[sync] 无需更新")
        return

    # 1) 新文档 → 分块 → 编码 → 补进 Chroma
    new_parents, new_children = [], []
    for d in missing:
        ps, cs = split_doc(d["doc_id"], d["text"])
        new_parents.extend(ps)
        new_children.extend(cs)
    print(f"[sync] 新增 {len(new_parents)} parents / {len(new_children)} children")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(config.KB_COLLECTION, metadata={"hnsw:space": "cosine"})
    model = get_embedder()
    embs = model.encode([c["text"] for c in new_children], batch_size=512,
                        show_progress_bar=False, normalize_embeddings=True)
    try:
        max_batch = client.get_max_batch_size()
    except Exception:
        max_batch = 4096
    add_batch = min(4096, max(1000, int(max_batch) - 512))
    for j in range(0, len(new_children), add_batch):
        col.add(
            ids=[c["child_id"] for c in new_children[j:j + add_batch]],
            documents=[c["text"] for c in new_children[j:j + add_batch]],
            embeddings=embs[j:j + add_batch].tolist(),
            metadatas=[{"doc_id": c["doc_id"], "parent_id": c["parent_id"]}
                       for c in new_children[j:j + add_batch]],
        )
    print(f"[sync] chroma 现有 {col.count()} children")

    # 2) 合并 parents.json
    for p in new_parents:
        parents[p["parent_id"]] = p["text"]
    with open(config.PARENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(parents, f, ensure_ascii=False)

    # 3) BM25 全量重建（重分词）
    print("[sync] 重建 BM25（全量重分词）...", flush=True)
    all_parents, all_children = [], []
    for d in docs:
        ps, cs = split_doc(d["doc_id"], d["text"])
        all_parents.extend(ps)
        all_children.extend(cs)
    tokenized = [jieba.lcut(c["text"]) for c in all_children]
    bm25 = BM25Okapi(tokenized)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "child_ids": [c["child_id"] for c in all_children],
            "child_parents": [c["parent_id"] for c in all_children],
        }, f)
    print(f"[sync] 完成：{len(all_parents)} parents / {len(all_children)} children")


if __name__ == "__main__":
    main()
