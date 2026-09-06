#!/usr/bin/env python3
"""用本地 BGE / 智谱 embedding 为 chunks 建向量索引（numpy）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from embedder import get_embedder  # noqa: E402
from llm import load_dotenv  # noqa: E402
from retriever import load_chunks  # noqa: E402

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, default=Path(__file__).resolve().parents[1] / "kb" / "chunks.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "kb")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="仅嵌入前 N 条（调试）")
    ap.add_argument("--backend", default=os.environ.get("MEISHI_EMBED_BACKEND", "local"),
                    help="local|zhipu|auto")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    if args.limit:
        chunks = chunks[: args.limit]
    texts = [c["text"] for c in chunks]
    embedder = get_embedder(args.backend)
    model_name = getattr(embedder, "model_name", None) or getattr(embedder, "model", "unknown")
    dim = getattr(embedder, "dimensions", None)
    print(f"embedding {len(texts)} chunks with {model_name} ...")
    vectors = embedder.embed_batched(texts, batch_size=args.batch_size)
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    mat = mat / norms
    if dim is None:
        dim = int(mat.shape[1])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = args.out_dir / "embeddings.npy"
    meta_path = args.out_dir / "embeddings_meta.json"
    np.save(npy_path, mat)
    meta = {
        "model": model_name,
        "backend": args.backend,
        "dimensions": dim,
        "n": len(chunks),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "normalized": True,
        "source_chunks": str(args.chunks),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(npy_path), "shape": list(mat.shape), "model": model_name}, ensure_ascii=False))


if __name__ == "__main__":
    main()
