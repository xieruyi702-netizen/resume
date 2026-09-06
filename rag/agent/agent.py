#!/usr/bin/env python3
"""美食汇食谱 Agent：检索 + 长短期记忆 + GLM 生成。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from llm import gen_client, load_dotenv
from memory import MemoryHub
from retriever import BM25Index, HybridIndex, build_index

load_dotenv()

SYSTEM = """你是「美食汇」烹饪助手。只能依据给定的 HowToCook 菜谱证据与记忆回答。
若证据不足，明确说不知道，不要编造步骤或用量。回答简洁，列出原料与关键步骤，并注明菜名来源。
若记忆中有忌口/偏好，生成时需遵守。"""


def maybe_rewrite(query: str) -> list[str]:
    q = query.strip()
    variants = [q]
    for noise in ("怎么做", "如何做", "的做法", "教我", "请问", "一下", "怎样做"):
        q2 = q.replace(noise, "").strip(" ？?。")
        if q2 and q2 != q:
            variants.append(q2)
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:3]


_FOOD_HINT = re.compile(
    r"(菜|饭|面|汤|粥|肉|鱼|虾|蛋|豆腐|炒|蒸|煮|炖|烤|卤|凉拌|原料|做法|步骤|怎么做|如何做|"
    r"鸡|鸭|牛|羊|猪|豆|菇|椒|醋|酱|油|盐|糖|蒜|姜|葱|米|粉|饺|饼)"
)


def retrieve_multi(index: HybridIndex | BM25Index, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
    merged: dict[str, tuple[dict, float]] = {}
    for q in maybe_rewrite(query):
        for chunk, score in index.search(q, top_k=top_k):
            cid = chunk["chunk_id"]
            if cid not in merged or score > merged[cid][1]:
                merged[cid] = (chunk, score)
    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def title_overlap(query: str, title: str) -> bool:
    """判断 query 是否指向该菜名。避免「煎饼」等通用字把库外菜误判为命中。"""
    if not title:
        return False
    if title in query:
        return True
    stop = set("的了吗呢是有和与在怎么如何做需要哪些请问教我一下帮我写段详细步骤用量原料家常正宗请给出")
    # 烹饪通用字：单独重合不算命中
    noise = set("肉蛋汤面饼饭菜丝片丁块汁酱油盐糖鱼虾鸡鸭炒煎炸煮炖蒸烤卤")
    q = set(re.findall(r"[\u4e00-\u9fff]", query)) - stop
    t = set(re.findall(r"[\u4e00-\u9fff]", title)) - stop
    if not t:
        return False
    inter = q & t
    sig = inter - noise
    if title in query:
        return True
    # 显著菜名用字至少 2 个，且总体重合足够
    if len(sig) >= 2 and len(inter) >= 3:
        return True
    if len(t) <= 3 and len(inter) >= len(t) and len(sig) >= 1:
        return True
    # 连续子串：覆盖菜名大部分
    for n in range(len(title), 2, -1):
        need = max(3, (len(title) * 2 + 2) // 3)
        if n < need:
            break
        for i in range(0, len(title) - n + 1):
            if title[i : i + n] in query:
                return True
    return False


def should_refuse(query: str, hits: list[tuple[dict, float]]) -> bool:
    """拒答门控。兼容 BM25 原始分（常 >2）与 hybrid RRF 分（常 0.01~0.04）。"""
    if not hits:
        return True
    top_chunk, top_score = hits[0]
    foodish = bool(_FOOD_HINT.search(query))
    overlapped = any(title_overlap(query, c.get("title", "")) for c, _ in hits[:3])
    # 菜名与 query 明显重合：视为命中，不拒
    if overlapped:
        return False
    # 非美食问法且无菜名重合
    if not foodish:
        return True
    if not re.search(r"[\u4e00-\u9fff]{2,}", query):
        return True
    # 无菜名重合：闭集知识库下默认拒答，避免库外菜被语义近邻「糊弄」成答案
    # （BM25 极高分仍可能是同字噪声，同样拒）
    return True


def template_answer(query: str, hits: list[tuple[dict, float]]) -> str:
    if not hits:
        return "知识库中未找到相关菜谱，无法回答。"
    best, score = hits[0]
    by_doc: dict[str, list[dict]] = {}
    for c, _ in hits:
        by_doc.setdefault(c["doc_id"], []).append(c)
    primary = by_doc[best["doc_id"]]
    parts = [f"根据美食汇知识库（HowToCook），推荐菜谱：**{best['title']}**（{best['category_zh']}）\n"]
    for label in ("原料", "用量", "步骤", "简介"):
        for c in primary:
            if c.get("section") == label or label in c["text"][:40]:
                body = c["text"].split("\n", 1)[-1].strip()
                if body:
                    parts.append(f"### {label}\n{body[:600]}")
                break
    cites = "、".join(sorted({c["title"] for c, _ in hits[:3]}))
    parts.append(f"\n来源菜谱：{cites}")
    return "\n\n".join(parts)


def glm_answer(query: str, hits: list[tuple[dict, float]], memory_block: str = "") -> str | None:
    try:
        client = gen_client()
    except Exception:
        return None
    evidence = "\n\n---\n\n".join(
        f"[{i+1}] {c['title']} / {c.get('section','')}\n{c['text'][:900]}"
        for i, (c, _) in enumerate(hits)
    )
    user = f"用户问题：{query}\n\n"
    if memory_block:
        user += f"{memory_block}\n\n"
    user += f"证据：\n{evidence}"
    try:
        return client.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1000,
        )
    except Exception as e:
        return f"[GLM 生成失败，回退模板] {e}"


def answer(
    index: HybridIndex | BM25Index,
    query: str,
    top_k: int = 5,
    memory: MemoryHub | None = None,
    use_llm: bool = True,
) -> dict:
    q = memory.session.rewrite_query(query) if memory else query
    mem_block = memory.inject_block(q) if memory else ""
    # 记忆偏好可拼进检索
    retrieval_q = q
    if memory and memory.session.working.get("preference"):
        retrieval_q = f"{q} {memory.session.working['preference']}"

    hits = retrieve_multi(index, retrieval_q, top_k=top_k)
    if should_refuse(q, hits):
        result = {
            "query": query,
            "resolved_query": q,
            "hits": [
                {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"], "score": round(s, 4)}
                for c, s in hits[:3]
            ],
            "answer": "问题与美食汇菜谱知识库无关或证据不足，已拒绝回答。",
            "refused": True,
            "memory_used": bool(mem_block),
        }
        if memory:
            memory.after_turn(query, result)
        return result

    ans = None
    if use_llm:
        raw = glm_answer(q, hits, mem_block)
        if raw and not raw.startswith("[GLM 生成失败"):
            ans = raw
        elif raw and raw.startswith("[GLM 生成失败"):
            ans = template_answer(q, hits) + "\n" + raw
    if not ans:
        ans = template_answer(q, hits)

    result = {
        "query": query,
        "resolved_query": q,
        "hits": [
            {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"], "score": round(s, 4)}
            for c, s in hits
        ],
        "answer": ans,
        "refused": False,
        "memory_used": bool(mem_block),
        "gen_model": gen_client().model if use_llm else "template",
    }
    if memory:
        memory.after_turn(query, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="美食汇食谱 RAG Agent")
    ap.add_argument("query", nargs="?", help="用户问题")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--agentic", action="store_true", help="启用 Agentic RAG 工具循环")
    ap.add_argument("--user", default="default", help="用户 id：短期/长期记忆按用户隔离")
    ap.add_argument("--session", default=None, help="同一用户下的会话 id（复用短期记忆）")
    args = ap.parse_args()
    index = build_index()
    memory = MemoryHub(user_id=args.user, session_id=args.session)

    def handle(q: str):
        if args.agentic:
            from agentic import run_agentic
            return run_agentic(q, index=index, memory=memory, use_llm=not args.no_llm)
        return answer(index, q, args.top_k, memory, use_llm=not args.no_llm)

    if args.interactive or not args.query:
        print(
            f"美食汇 Agent user={memory.user_id} session={memory.session.session_id} "
            f"agentic={args.agentic}（输入 q 退出）"
        )
        while True:
            q = input("> ").strip()
            if not q or q.lower() in {"q", "quit", "exit"}:
                break
            print(json.dumps(handle(q), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(handle(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
