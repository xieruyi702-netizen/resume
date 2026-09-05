"""构建知识库索引入口。用法：
  uv run python scripts/build_knowledge.py          # 增量（已有则跳过）
  uv run python scripts/build_knowledge.py --force  # 重建
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.indexer import build_index  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    corpus = Path(__file__).resolve().parents[1] / "data" / "kb_docs.jsonl"
    build_index(corpus, force=args.force)
