#!/usr/bin/env python3
"""检索评测 + GLM 生成 + 另一免费 GLM 做 LLM-as-Judge。

支持题型：qa（库内）/ missing（库外菜）/ ood（非食谱）。
可用 --judge-sample 对生成题做分层抽样评测。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from agent import answer, retrieve_multi  # noqa: E402
from llm import gen_client, judge_client, load_dotenv  # noqa: E402
from memory import MemoryHub  # noqa: E402
from retriever import build_index  # noqa: E402

load_dotenv()

JUDGE_PROMPT = """你是严格的中文菜谱问答评测员。根据【参考标准】判断【助手回答】质量。
只输出 JSON，不要 markdown：
{"faithfulness":0或1,"relevance":0或1,"completeness":0或1,"no_hallucination":0或1,"score":0到1的小数,"reason":"一句话"}
定义：
- faithfulness: 答案关键事实是否可由参考支持（步骤/原料不编造）
- relevance: 是否回答了用户问题（含：该拒答时正确拒答也算相关）
- completeness: 是否覆盖问题所需的关键信息（原料题给原料、步骤题给步骤；拒答题说清未收录/超范围即可）
- no_hallucination: 没有明显臆造菜名/用量/步骤
- score: 综合分 = 四项平均

特殊规则：
1) 参考标明「库中没有/应拒答」：助手明确拒答或声明未收录 → 四项均 1，score=1；若编造完整菜谱 → no_hallucination=0，completeness=0，score≤0.25。
2) 参考标明「与菜谱无关」：助手拒答/声明超范围 → 四项均 1；若认真回答编程/金融等 → relevance/completeness 可给，但 no_hallucination 与 faithfulness 按是否越权瞎答，通常 score≤0.5。
3) 库内菜谱题却拒答：faithfulness=1，relevance=0，completeness=0，no_hallucination=1，score=0.5。
"""


def load_cases(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eval_retrieval(index, cases: list[dict], k: int = 5) -> dict:
    hits_at = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    recall5 = 0.0
    n = 0
    for c in cases:
        if c.get("type") != "qa":
            continue
        n += 1
        gold = set(c["relevant_doc_ids"])
        ranked = retrieve_multi(index, c["query"], top_k=k)
        docs = [h[0]["doc_id"] for h in ranked]
        rr = 0.0
        for i, d in enumerate(docs):
            if d in gold:
                rr = 1.0 / (i + 1)
                break
        mrr += rr
        for kk in (1, 3, 5):
            if any(d in gold for d in docs[:kk]):
                hits_at[kk] += 1
        if gold:
            recall5 += len(gold & set(docs[:5])) / len(gold)
    return {
        "n": n,
        "Hit@1": round(hits_at[1] / n, 4) if n else 0,
        "Hit@3": round(hits_at[3] / n, 4) if n else 0,
        "Hit@5": round(hits_at[5] / n, 4) if n else 0,
        "MRR": round(mrr / n, 4) if n else 0,
        "Recall@5": round(recall5 / n, 4) if n else 0,
    }


def parse_judge(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"score": 0.0, "raw": text[:200], "parse_error": True}
    try:
        obj = json.loads(m.group())
        for k in ("faithfulness", "relevance", "completeness", "no_hallucination"):
            if k in obj:
                obj[k] = int(bool(obj[k])) if not isinstance(obj[k], (int, float)) else int(obj[k] >= 0.5)
        if "score" not in obj:
            vals = [obj.get(k, 0) for k in ("faithfulness", "relevance", "completeness", "no_hallucination")]
            obj["score"] = sum(vals) / 4
        obj["score"] = float(obj["score"])
        obj["parse_error"] = False
        return obj
    except Exception:
        return {"score": 0.0, "raw": text[:200], "parse_error": True}


def judge_one(jclient, query: str, answer_text: str, reference: str, case_type: str) -> dict:
    if reference:
        ref = reference[:1600]
    elif case_type in ("ood", "missing"):
        ref = "（无参考正文；正确行为是拒答或声明知识库未收录/超范围）"
    else:
        ref = "（无参考）"
    user = f"题型：{case_type}\n用户问题：{query}\n\n参考标准：\n{ref}\n\n助手回答：\n{answer_text[:1500]}"
    raw = jclient.chat(
        [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=400,
    )
    return parse_judge(raw)


def stratified_sample(cases: list[dict], n: int, seed: int) -> list[dict]:
    """按 type 分层抽样，尽量覆盖 qa / missing / ood。"""
    if n <= 0 or n >= len(cases):
        return list(cases)
    rng = random.Random(seed)
    by_t: dict[str, list] = {}
    for c in cases:
        by_t.setdefault(c.get("type") or "qa", []).append(c)
    # 目标比例：qa 60% / missing 20% / ood 20%
    want = {
        "qa": max(1, int(round(n * 0.6))),
        "missing": max(1, int(round(n * 0.2))),
        "ood": max(1, int(round(n * 0.2))),
    }
    # 修正总数
    while sum(want.values()) > n:
        for k in ("qa", "missing", "ood"):
            if want[k] > 1 and sum(want.values()) > n:
                want[k] -= 1
    while sum(want.values()) < n:
        want["qa"] += 1

    picked: list[dict] = []
    for t, k in want.items():
        pool = list(by_t.get(t) or [])
        rng.shuffle(pool)
        take = pool[: min(k, len(pool))]
        picked.extend(take)
        # 不足从 qa 补
        if len(take) < k:
            extra_need = k - len(take)
            rest = [c for c in (by_t.get("qa") or []) if c not in picked]
            rng.shuffle(rest)
            picked.extend(rest[:extra_need])

    # 仍不足则从全体补
    if len(picked) < n:
        rest = [c for c in cases if c not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    rng.shuffle(picked)
    return picked[:n]


def is_refuse_ok(answer_text: str, refused: bool | None) -> bool:
    if refused:
        return True
    text = answer_text or ""
    keys = (
        "拒绝", "无关", "未收录", "没有这道", "不在知识库", "超出", "无法提供",
        "抱歉", "不知道", "证据不足", "未找到", "没有相关",
    )
    return any(k in text for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=Path(__file__).resolve().parent / "testset.jsonl")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "last_metrics.json")
    ap.add_argument("--details", type=Path, default=Path(__file__).resolve().parent / "last_details.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="仅跑前 N 条（0=全部；作用于原始集）")
    ap.add_argument("--judge-sample", type=int, default=0,
                    help="生成+Judge 分层抽样条数（0=对全部跑生成+Judge；>0 时检索仍全量）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-llm", action="store_true", help="生成用模板，仍可用 Judge")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()

    index = build_index()
    cases = load_cases(args.testset)
    if args.limit:
        cases = cases[: args.limit]

    retrieval = eval_retrieval(index, cases)
    print(json.dumps({"retrieval": retrieval}, ensure_ascii=False, indent=2))

    if args.retrieval_only:
        metrics = {"kb": "HowToCook", "retrieval": retrieval, "n_cases": len(cases)}
        args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    gen_cases = stratified_sample(cases, args.judge_sample, args.seed) if args.judge_sample else list(cases)
    print(json.dumps({
        "gen_judge_n": len(gen_cases),
        "by_type": {
            t: sum(1 for c in gen_cases if c.get("type") == t)
            for t in ("qa", "missing", "ood")
        },
    }, ensure_ascii=False))

    gmodel = gen_client().model
    jmodel = judge_client().model
    jclient = None if args.no_judge else judge_client()

    details = []
    refuse_ok = {"missing": [0, 0], "ood": [0, 0]}  # ok, n
    judge_scores = []
    judge_by_type: dict[str, list] = {"qa": [], "missing": [], "ood": []}
    dims = {"faithfulness": [], "relevance": [], "completeness": [], "no_hallucination": []}

    for i, c in enumerate(gen_cases):
        memory = MemoryHub(user_id=f"eval-{c['id']}", session_id=f"eval-{c['id']}", persist_session=False)
        r = answer(index, c["query"], top_k=5, memory=memory, use_llm=not args.no_llm)
        ctype = c.get("type") or "qa"
        row = {
            "id": c["id"],
            "type": ctype,
            "query": c["query"],
            "refused": r.get("refused"),
            "answer": r.get("answer", "")[:2000],
            "hits": r.get("hits", [])[:5],
            "has_reference": bool(c.get("reference")),
        }
        if ctype in refuse_ok:
            refuse_ok[ctype][1] += 1
            if is_refuse_ok(r.get("answer", ""), r.get("refused")):
                refuse_ok[ctype][0] += 1

        if jclient is not None:
            try:
                j = judge_one(
                    jclient,
                    c["query"],
                    r.get("answer", ""),
                    c.get("reference", ""),
                    case_type=ctype,
                )
                row["judge"] = j
                if not j.get("parse_error"):
                    judge_scores.append(j["score"])
                    judge_by_type.setdefault(ctype, []).append(j["score"])
                    for k in dims:
                        if k in j:
                            dims[k].append(j[k])
            except Exception as e:
                row["judge"] = {"error": str(e)[:200]}
            time.sleep(args.sleep)

        details.append(row)
        print(
            f"[{i+1}/{len(gen_cases)}] {c['id']} type={ctype} refused={r.get('refused')} "
            f"judge={row.get('judge', {}).get('score', row.get('judge', {}).get('error', '-'))}"
        )

    with args.details.open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def rate(pair: list[int]) -> float | None:
        ok, n = pair
        return round(ok / n, 4) if n else None

    gen_metrics = {
        "gen_model": gmodel if not args.no_llm else "template",
        "judge_model": None if args.no_judge else jmodel,
        "judge_sample_n": len(gen_cases),
        "judge_sample_seed": args.seed if args.judge_sample else None,
        "missing_refuse_rate": rate(refuse_ok["missing"]),
        "ood_refuse_rate": rate(refuse_ok["ood"]),
        "n_judged": len(judge_scores),
        "glm_judge_avg_score": round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None,
        "glm_judge_by_type": {
            t: round(sum(v) / len(v), 4) if v else None for t, v in judge_by_type.items()
        },
        "glm_judge_dims": {
            k: round(sum(v) / len(v), 4) if v else None for k, v in dims.items()
        },
    }
    metrics = {
        "kb": "HowToCook",
        "retrieval": retrieval,
        "generation": gen_metrics,
        "n_cases": len(cases),
        "testset_mix": {
            "qa": sum(1 for c in cases if c.get("type") == "qa"),
            "missing": sum(1 for c in cases if c.get("type") == "missing"),
            "ood": sum(1 for c in cases if c.get("type") == "ood"),
        },
    }
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
