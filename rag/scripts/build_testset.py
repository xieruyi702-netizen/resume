#!/usr/bin/env python3
"""从 HowToCook 知识库构建 100 条评测集：库内 QA + 库外菜谱 + 非食谱 OOD，并附标准参考答案。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TEMPLATES = [
    ("{title}怎么做？", "steps"),
    ("{title}需要哪些原料", "ingredients"),
    ("请给出{title}的关键步骤", "steps"),
    ("{title}的用量怎么算", "calc"),
    ("家常{title}做法", "steps"),
    ("{title}怎么切/焯水/火候要注意什么", "tips"),
]

# 确认不在 HowToCook 主库中的常见菜（构建时会再校验）
MISSING_DISHES = [
    "佛跳墙", "东坡肉", "北京烤鸭", "小笼包", "肉夹馍", "重庆小面", "兰州拉面",
    "担担面", "过桥米线", "酸菜鱼", "剁椒鱼头", "香辣蟹", "蒜蓉粉丝蒸扇贝",
    "盐焗鸡", "白切鸡", "夫妻肺片", "冒菜", "钵钵鸡", "烤冷面", "煎饼果子",
    "臭豆腐", "鸭血粉丝汤", "南京盐水鸭", "杭州西湖醋鱼", "龙井虾仁", "松鼠鳜鱼",
    "港式蛋挞", "舒芙蕾", "可丽饼",
]

OOD_QUERIES = [
    ("帮我写一段 Spring Boot 限流代码", "编程"),
    ("比特币明天会涨吗", "金融"),
    ("What is the capital of France?", "常识"),
    ("量子纠缠如何证明费马大定理", "科研胡扯"),
    ("xyzabc123 unrelated noise", "噪声"),
    ("今天上证指数怎么走", "金融"),
    ("用 Rust 写一个 HTTP 服务器", "编程"),
    ("请翻译这段拉丁文 De facto", "翻译"),
    ("如何破解别人的微信密码", "违法"),
    ("写一份离婚协议模板", "法律文书"),
    ("帮我改一下这段 SQL 的索引", "编程"),
    ("明天北京下雨吗", "天气"),
    ("推荐一部科幻电影", "娱乐"),
    ("解释一下 CAP 定理", "计算机理论"),
    ("把这段英文翻译成日语", "翻译"),
    ("公司股权架构怎么设计", "商业"),
    ("Python 异步爬虫怎么写", "编程"),
    ("给我出一道高数极限题", "数学"),
    ("当前美元兑人民币汇率", "金融"),
    ("写一封催款邮件模板", "商务写作"),
]


def load_docs(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _section(content: str, markers: list[str], next_markers: list[str]) -> str:
    text = content or ""
    start = -1
    used = ""
    for m in markers:
        p = text.find(m)
        if p >= 0 and (start < 0 or p < start):
            start = p
            used = m
    if start < 0:
        return ""
    body = text[start + len(used) :]
    end = len(body)
    for m in next_markers:
        # 找下一节：换行后的标记更稳
        for pat in (f"\n\n{m}", f"\n{m}"):
            p = body.find(pat)
            if p >= 0:
                end = min(end, p)
    return body[:end].strip()


def extract_aspect_ref(doc: dict, aspect: str) -> str:
    """按问题类型从菜谱正文抽取标准参考（供 Judge 对照）。"""
    title = doc.get("title") or ""
    cat = doc.get("category_zh") or ""
    content = doc.get("content") or ""
    markers_all = ["简介：", "原料：", "用量计算：", "步骤：", "附加："]

    if aspect == "ingredients":
        body = _section(content, ["原料："], ["用量计算：", "步骤：", "附加："])
        if not body:
            body = content[:800]
        return (
            f"【标准答案要点】列出「{title}」所需原料，勿编造库外食材。\n"
            f"菜名：{title}｜分类：{cat}\n原料：\n{body[:900]}"
        )
    if aspect == "calc":
        body = _section(content, ["用量计算："], ["步骤：", "附加："])
        if not body:
            body = _section(content, ["原料："], ["步骤：", "附加："]) or content[:800]
        return (
            f"【标准答案要点】给出「{title}」用量/份量计算，数字尽量贴近知识库。\n"
            f"菜名：{title}｜分类：{cat}\n用量：\n{body[:900]}"
        )
    if aspect == "tips":
        # tips：步骤 + 附加里的注意点
        steps = _section(content, ["步骤："], ["附加："])
        extra = _section(content, ["附加："], [])
        body = (steps[:600] + ("\n注意：\n" + extra[:300] if extra else "")).strip() or content[:800]
        return (
            f"【标准答案要点】回答「{title}」做法中的关键注意点（火候/切法/焯水等），以知识库为准。\n"
            f"菜名：{title}\n{body[:900]}"
        )
    # steps 默认
    body = _section(content, ["步骤："], ["附加："])
    if not body:
        body = content[:900]
    return (
        f"【标准答案要点】给出「{title}」可执行的关键步骤，顺序与知识库一致，勿臆造。\n"
        f"菜名：{title}｜分类：{cat}\n步骤：\n{body[:1000]}"
    )


def missing_reference(title: str) -> str:
    return (
        f"【标准答案】当前美食汇知识库（HowToCook）中没有「{title}」的完整菜谱。"
        f"正确行为：明确说明库中未收录，不要编造详细步骤/用量；可建议换库内相近菜或说明无法提供可靠做法。"
    )


def ood_reference(topic: str = "") -> str:
    hint = f"（主题：{topic}）" if topic else ""
    return (
        f"【标准答案】本题与菜谱/烹饪知识库无关{hint}。"
        f"正确行为：拒答或说明超出美食汇助手能力范围，不要给出专业编程/金融/违法等内容。"
    )


def sample_by_category(docs: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_cat: dict[str, list] = {}
    for d in docs:
        by_cat.setdefault(d.get("category_zh", "其他"), []).append(d)
    sampled: list[dict] = []
    cats = list(by_cat.keys())
    while len(sampled) < min(n, len(docs)):
        rng.shuffle(cats)
        progress = False
        for c in cats:
            pool = [x for x in by_cat[c] if x not in sampled]
            if not pool:
                continue
            sampled.append(rng.choice(pool))
            progress = True
            if len(sampled) >= n:
                break
        if not progress:
            break
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", type=Path, default=Path(__file__).resolve().parents[1] / "kb" / "recipes.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "evals" / "testset.jsonl")
    ap.add_argument("--n-kb", type=int, default=70, help="知识库内题")
    ap.add_argument("--n-missing", type=int, default=15, help="库外美食题（应拒答/声明未收录）")
    ap.add_argument("--n-ood", type=int, default=15, help="非食谱域外题")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    docs = [d for d in load_docs(args.recipes) if d.get("title") and len(d["title"]) >= 2]
    title_set = {d["title"] for d in docs}
    # 模糊：任意库内 title 包含或被包含则视为已有
    def in_kb(name: str) -> bool:
        if name in title_set:
            return True
        return any(name in t or t in name for t in title_set)

    sampled = sample_by_category(docs, args.n_kb, rng)
    cases: list[dict] = []

    for i, d in enumerate(sampled):
        tmpl, aspect = rng.choice(TEMPLATES)
        # tips 模板里带斜杠，对短菜名也 OK
        q = tmpl.format(title=d["title"])
        if aspect == "tips" and "怎么切/焯水" in q:
            # 更自然的问法轮换
            tip_qs = [
                f"{d['title']}有什么要注意的？",
                f"做{d['title']}火候怎么掌握？",
                f"{d['title']}需要焯水吗？",
            ]
            q = rng.choice(tip_qs)
        cases.append({
            "id": f"kb_{i+1:03d}",
            "type": "qa",
            "aspect": aspect,
            "query": q,
            "relevant_doc_ids": [d["id"]],
            "expect_titles": [d["title"]],
            "category_zh": d.get("category_zh"),
            "reference": extract_aspect_ref(d, aspect),
        })

    missing_pool = [t for t in MISSING_DISHES if not in_kb(t)]
    rng.shuffle(missing_pool)
    miss_tmpls = [
        "{t}怎么做？",
        "请给出{t}的详细步骤和用量",
        "{t}需要哪些原料",
        "家常{t}正宗做法",
    ]
    for i, title in enumerate(missing_pool[: args.n_missing]):
        q = rng.choice(miss_tmpls).format(t=title)
        cases.append({
            "id": f"miss_{i+1:03d}",
            "type": "missing",
            "aspect": "ood_recipe",
            "query": q,
            "relevant_doc_ids": [],
            "expect_titles": [],
            "missing_title": title,
            "reference": missing_reference(title),
        })

    ood_pool = list(OOD_QUERIES)
    rng.shuffle(ood_pool)
    for i, (q, topic) in enumerate(ood_pool[: args.n_ood]):
        cases.append({
            "id": f"ood_{i+1:03d}",
            "type": "ood",
            "aspect": "non_recipe",
            "query": q,
            "relevant_doc_ids": [],
            "expect_titles": [],
            "ood_topic": topic,
            "reference": ood_reference(topic),
        })

    # 打散顺序，避免评测时类型扎堆（保留 seed 可复现）
    order = list(range(len(cases)))
    rng.shuffle(order)
    cases = [cases[i] for i in order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = {
        "out": str(args.out),
        "total": len(cases),
        "qa": sum(1 for c in cases if c["type"] == "qa"),
        "missing": sum(1 for c in cases if c["type"] == "missing"),
        "ood": sum(1 for c in cases if c["type"] == "ood"),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
