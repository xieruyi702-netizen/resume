"""把 CRUD-RAG 转成项目格式：
  data/kb_docs.jsonl   全量知识库文档（80k 新闻去重后） {doc_id, text}
  evals/golden.jsonl   最小黄金评测集（默认 8×单跳 + 3×双跳 + 2×拒答）

doc_id = "doc_" + md5(正文规范化前 120 字)，QA 的 news 字段用同一哈希
映射回语料（QA 与语料的文本前缀格式略有差异，故从"正文："之后取哈希）；
个别对不上的 QA 来源文档会以 qa_ 前缀补进知识库，保证 expected_sources 必在库中。
"""
import argparse
import glob
import hashlib
import json
import random
import re
from pathlib import Path


def norm_body(text: str, n: int = 120) -> str:
    """取"正文："之后的正文，去空白，截前 n 字——QA 与语料的公共可比对段。"""
    if "正文：" in text:
        text = text.split("正文：", 1)[1]
    return re.sub(r"\s+", "", text)[:n]


def body_hash(text: str) -> str:
    return hashlib.md5(norm_body(text).encode("utf-8")).hexdigest()


def extract_keywords(answer: str) -> list[str]:
    """从参考答案自动提取可校验关键词：书名号标题优先，其次数字，兜底最长汉字串。"""
    kws = re.findall(r"《[^》]{2,30}》", answer)
    nums = re.findall(r"[0-9]+(?:\.[0-9]+)?%?", answer)
    kws += nums[:2]
    if not kws:
        runs = re.findall(r"[\u4e00-\u9fff]{4,}", answer)
        if runs:
            kws = [max(runs, key=len)[:8]]
    out, seen = [], set()
    for kw in kws:
        if kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out[:3]


REFUSAL_CASES = [
    {
        "id": "refuse-01",
        "type": "refusal",
        "question": "2026年冬奥会新增了哪些比赛项目？",
        "expected_sources": [],
        "expected_keywords": [],
        "expect_refusal": True,
        "reference_answer": "知识库为2023年新闻语料，不含该信息，应拒答",
    },
    {
        "id": "refuse-02",
        "type": "refusal",
        "question": "本知识库的检索系统用的是哪家公司的向量数据库产品？",
        "expected_sources": [],
        "expected_keywords": [],
        "expect_refusal": True,
        "reference_answer": "知识库内容为新闻正文，不含系统自身信息，应拒答",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/CRUD_RAG", help="CRUD_RAG 克隆目录")
    ap.add_argument("--n1", type=int, default=8, help="单跳（1doc）case 数")
    ap.add_argument("--n2", type=int, default=3, help="双跳（2docs）case 数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)

    # ---- 1) 全量语料去重 → kb_docs.jsonl，并建 哈希→doc_id 映射 ----
    corpus_path = data_dir / "kb_docs.jsonl"
    split_path = Path(args.repo) / "data" / "crud_split" / "split_merged.json"
    body2id: dict[str, str] = {}
    n_raw = n_dup = 0
    with open(corpus_path, "w", encoding="utf-8") as out:
        files = sorted(glob.glob(str(Path(args.repo) / "data" / "80000_docs" / "*")))
        for fp in files:
            for line in open(fp, encoding="utf-8"):
                text = line.strip()
                if not text:
                    continue
                n_raw += 1
                h = body_hash(text)
                if h in body2id:
                    n_dup += 1
                    continue
                doc_id = f"doc_{h[:12]}"
                body2id[h] = doc_id
                out.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    print(f"[convert] 语料 {n_raw} 行 → 去重后 {len(body2id)} 篇（重复 {n_dup}）→ {corpus_path}")

    # ---- 2) 抽样 QA → golden.jsonl ----
    data = json.load(open(split_path, encoding="utf-8"))
    rng = random.Random(args.seed)
    cases = []
    extras = []  # 语料里没匹配上的 QA 来源文档，补进知识库

    def doc_id_for(news_text: str) -> str:
        h = body_hash(news_text)
        if h in body2id:
            return body2id[h]
        doc_id = f"qa_{h[:12]}"
        if all(d["doc_id"] != doc_id for d in extras):
            extras.append({"doc_id": doc_id, "text": news_text.strip()})
        return doc_id

    for sub, n, case_type in [("questanswer_1doc", args.n1, "single_hop"),
                              ("questanswer_2docs", args.n2, "multi_hop")]:
        pool = data[sub]
        for item in rng.sample(pool, min(n, len(pool))):
            srcs = [doc_id_for(item[k]) for k in item if k.startswith("news")]
            cases.append({
                "id": f"{case_type.split('_')[0]}-{item['ID'][:8]}-{len(cases) + 1:02d}",
                "type": case_type,
                "question": item["questions"],
                "expected_sources": srcs,
                "expected_keywords": extract_keywords(item["answers"]),
                "expect_refusal": False,
                "reference_answer": item["answers"],
            })

    cases += REFUSAL_CASES
    golden_path = root / "evals" / "golden.jsonl"
    with open(golden_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    if extras:
        with open(corpus_path, "a", encoding="utf-8") as out:
            for d in extras:
                out.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[convert] golden {len(cases)} 条（含拒答 {len(REFUSAL_CASES)}）→ {golden_path}")
    print(f"[convert] QA 来源文档补入知识库 {len(extras)} 篇")
    for c in cases:
        mark = "✓" if c["expected_sources"] or c["expect_refusal"] else "?"
        print(f"  {mark} {c['id']}  kw={c['expected_keywords']}  srcs={len(c['expected_sources'])}")


if __name__ == "__main__":
    main()
