"""构建索引：child 向量入 Chroma（持久化）+ jieba 分词建 BM25。

幂等：已存在则跳过，--force 重建。产物：
  data/kb/chroma/        child 向量库（collection: kb_children）
  data/kb/parents.json   parent_id -> 全文
  data/kb/bm25.pkl       BM25 索引 + child_ids + child→parent 映射
"""
import json
import pickle

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src import config
from src.ingest.chunker import split_doc


def build_index(corpus_path, force: bool = False) -> None:
    if not force and config.BM25_PATH.exists() and config.PARENTS_PATH.exists():
        print("[index] 索引已存在，跳过（--force 可重建）")
        return

    docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
    parents, children = [], []
    for d in docs:
        ps, cs = split_doc(d["doc_id"], d["text"])
        parents.extend(ps)
        children.extend(cs)
    print(f"[index] {len(docs)} 篇文档 → {len(parents)} parents / {len(children)} children")

    config.KB_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PARENTS_PATH, "w", encoding="utf-8") as f:
        json.dump({p["parent_id"]: p["text"] for p in parents}, f, ensure_ascii=False)

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(config.EMBED_MODEL, device=device)
    if device == "mps":
        model.half()  # fp16 在 MPS 上约 1.6x 提速
    print(f"[index] 编码 {len(children)} 个 child 向量（{config.EMBED_MODEL} @ {device}）...")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_or_create_collection(config.KB_COLLECTION, metadata={"hnsw:space": "cosine"})

    # 分块「编码→入库」，避免一次性大数组占内存，也方便观察进度
    # chroma 对单次 add 有条数上限（本版本 5461），动态取值并取安全下界
    try:
        max_batch = client.get_max_batch_size()
    except Exception:
        max_batch = 4096
    add_batch = min(4096, max(1000, int(max_batch) - 512))

    chunk = 20000
    for start in range(0, len(children), chunk):
        seg = children[start:start + chunk]
        embs = model.encode([c["text"] for c in seg], batch_size=512,
                            show_progress_bar=False, normalize_embeddings=True)
        ids = [c["child_id"] for c in seg]
        metas = [{"doc_id": c["doc_id"], "parent_id": c["parent_id"]} for c in seg]
        docs = [c["text"] for c in seg]
        for j in range(0, len(seg), add_batch):
            col.add(ids=ids[j:j + add_batch], documents=docs[j:j + add_batch],
                    embeddings=embs[j:j + add_batch].tolist(), metadatas=metas[j:j + add_batch])
        print(f"[index] chroma {min(start + chunk, len(children))}/{len(children)}", flush=True)

    print("[index] jieba 分词 + BM25 ...", flush=True)
    texts = [c["text"] for c in children]
    tokenized = [jieba.lcut(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "child_ids": [c["child_id"] for c in children],
            "child_parents": [c["parent_id"] for c in children],
        }, f)
    print("[index] 完成")
