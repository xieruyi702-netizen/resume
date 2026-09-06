#!/usr/bin/env python3
"""将 HowToCook dishes/**/*.md 解析为美食汇 RAG 知识库 JSONL。

来源: https://github.com/Anduin2017/HowToCook (Unlicense)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CATEGORY_ZH = {
    "aquatic": "水产",
    "breakfast": "早餐",
    "condiment": "酱料",
    "dessert": "甜品",
    "drink": "饮料",
    "meat_dish": "荤菜",
    "semi-finished": "半成品",
    "soup": "汤粥",
    "staple": "主食",
    "vegetable_dish": "素菜",
    "template": "模板",
}


def extract_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+?)(?:的做法)?\s*$", text, re.M)
    if m:
        return m.group(1).strip()
    return fallback


def extract_section(text: str, heading: str) -> str:
    pat = rf"^##\s+{re.escape(heading)}\s*$"
    m = re.search(pat, text, re.M)
    if not m:
        return ""
    start = m.end()
    n = re.search(r"^##\s+", text[start:], re.M)
    end = start + n.start() if n else len(text)
    return text[start:end].strip()


def difficulty_stars(text: str) -> int:
    m = re.search(r"预估烹饪难度：([★☆]+)", text)
    return m.group(1).count("★") if m else 0


def parse_md(path: Path, dishes_root: Path) -> dict | None:
    rel = path.relative_to(dishes_root)
    parts = rel.parts
    if not parts:
        return None
    category = parts[0]
    if category == "template":
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text or len(text) < 40:
        return None
    title = extract_title(text, path.stem)
    ingredients = extract_section(text, "必备原料和工具")
    calc = extract_section(text, "计算")
    steps = extract_section(text, "操作")
    extra = extract_section(text, "附加内容")
    # 简介：标题后到第一个 ## 之前
    intro_m = re.search(r"^#\s+.+\n+([\s\S]*?)(?=^##\s+)", text, re.M)
    intro = intro_m.group(1).strip() if intro_m else ""

    doc_id = str(rel.with_suffix("")).replace("\\", "/")
    category_zh = CATEGORY_ZH.get(category, category)
    body_parts = [
        f"菜名：{title}",
        f"分类：{category_zh}",
        f"简介：{intro}" if intro else "",
        f"原料：\n{ingredients}" if ingredients else "",
        f"用量计算：\n{calc}" if calc else "",
        f"步骤：\n{steps}" if steps else "",
        f"附加：\n{extra}" if extra else "",
    ]
    content = "\n\n".join(p for p in body_parts if p)
    return {
        "id": doc_id,
        "title": title,
        "category": category,
        "category_zh": category_zh,
        "difficulty": difficulty_stars(text),
        "source": "HowToCook",
        "source_path": str(rel),
        "content": content,
        "raw_md": text,
    }


def chunk_doc(doc: dict, max_chars: int = 800) -> list[dict]:
    """按章节优先切块，过长再按段落滑动。"""
    sections = []
    for label, key in [
        ("简介", None),
        ("原料", "原料："),
        ("用量", "用量计算："),
        ("步骤", "步骤："),
        ("附加", "附加："),
    ]:
        if key is None:
            # 取简介段
            m = re.search(r"简介：([\s\S]*?)(?=\n\n原料：|\n\n用量|\n\n步骤|\Z)", doc["content"])
            text = (m.group(1).strip() if m else "")
        else:
            m = re.search(rf"{re.escape(key)}\n([\s\S]*?)(?=\n\n(?:原料|用量计算|步骤|附加)：|\Z)", doc["content"])
            text = (m.group(1).strip() if m else "")
        if text:
            sections.append((label, text))

    chunks = []
    if not sections:
        sections = [("全文", doc["content"])]

    for i, (label, text) in enumerate(sections):
        prefix = f"{doc['title']}｜{doc['category_zh']}｜{label}\n"
        if len(prefix) + len(text) <= max_chars:
            chunks.append({
                "chunk_id": f"{doc['id']}#{i}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "category_zh": doc["category_zh"],
                "section": label,
                "text": prefix + text,
            })
            continue
        # 滑动切分
        step = max_chars - len(prefix) - 20
        for j in range(0, len(text), step):
            piece = text[j : j + step]
            chunks.append({
                "chunk_id": f"{doc['id']}#{i}.{j}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "category_zh": doc["category_zh"],
                "section": label,
                "text": prefix + piece,
            })
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "HowToCook" / "dishes",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "kb",
    )
    ap.add_argument("--max-chars", type=int, default=800)
    args = ap.parse_args()

    docs = []
    for path in sorted(args.src.rglob("*.md")):
        doc = parse_md(path, args.src)
        if doc:
            docs.append(doc)

    chunks = []
    for d in docs:
        chunks.extend(chunk_doc(d, args.max_chars))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs_path = args.out_dir / "recipes.jsonl"
    chunks_path = args.out_dir / "chunks.jsonl"
    meta_path = args.out_dir / "meta.json"

    with docs_path.open("w", encoding="utf-8") as f:
        for d in docs:
            slim = {k: v for k, v in d.items() if k != "raw_md"}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_cat: dict[str, int] = {}
    for d in docs:
        by_cat[d["category_zh"]] = by_cat.get(d["category_zh"], 0) + 1
    meta = {
        "source": "https://github.com/Anduin2017/HowToCook",
        "license": "Unlicense",
        "docs": len(docs),
        "chunks": len(chunks),
        "by_category": by_cat,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
