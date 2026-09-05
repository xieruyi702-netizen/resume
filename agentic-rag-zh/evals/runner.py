"""评测 runner：两层指标一次跑完。

  Stage A 检索层（离线、零 LLM 成本）：Hit@5 / MRR / Recall@5 / nDCG@5
  Stage B 答案层（需 LLM）：关键词命中率、拒答准确率、平均步数、token 估算；
          失败 case 根因归因后写入失败记忆库（--memory 控制的是"召回注入"，
          写入始终开启，这样 off→on 两次运行即构成记忆消融实验）。
"""
import json
import time
from pathlib import Path

from evals import metrics
from src import config
from src.agent.agent import Agent
from src.memory.failure_store import FailureMemory, attribute_failure
from src.retrieval.retriever import get_retriever


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in open(path, encoding="utf-8")]
    return cases[:limit] if limit else cases


def run_eval(tag: str, *, with_llm: bool = False, memory_on: bool = False,
             use_judge: bool = False, limit: int | None = None, top_k: int = 5,
             mode: str = "react") -> dict:
    started = time.time()
    cases = load_cases(config.GOLDEN_PATH, limit)
    retriever = get_retriever()
    llm = None
    agent = None
    memory = None
    judge_llm = None
    if with_llm:
        from src.llm.provider import get_llm
        llm = get_llm()
        memory = FailureMemory()
        if mode == "pae":
            from src.agent.graph_pae import PlanExecuteAgent
            agent = PlanExecuteAgent(llm, retriever, memory=memory)
        else:
            agent = Agent(llm, retriever, memory=memory)
        if use_judge:
            from evals.judge import get_judge_llm
            judge_llm = get_judge_llm()
            if judge_llm is None:
                print("[runner] 未配置 JUDGE_API_KEY，本次退回纯规则评测")

    per_case = []
    hits = mrr_sum = rec_sum = ndcg_sum = 0.0
    n_with_src = 0
    kw_ok = refusal_ok_n = n_ans = n_ref = 0
    judge_ok = n_judged = 0
    rel_sum = 0.0
    effective_ok = 0
    steps_sum = chars_sum = 0

    for i, case in enumerate(cases, 1):
        expected = case["expected_sources"]
        chunks = retriever.retrieve(case["question"], k=top_k)
        sources = [c.doc_id for c in chunks]
        hit = metrics.hit_at_k(sources, expected, top_k)
        rr = metrics.reciprocal_rank(sources, expected)
        rec = metrics.recall_at_k(sources, expected, top_k)
        ndcg = metrics.ndcg_at_k(sources, expected, top_k)
        if expected:
            n_with_src += 1
            hits += hit
            mrr_sum += rr
            rec_sum += rec
            ndcg_sum += ndcg

        row = {
            "id": case["id"],
            "type": case["type"],
            "mode": mode,
            "question": case["question"][:50],
            "hit": hit, "rr": round(rr, 3), "recall": round(rec, 3), "ndcg": round(ndcg, 3),
            "retrieved": sources[:top_k],
        }

        if with_llm:
            trace = agent.run(case["question"], memory_on=memory_on)
            if case["expect_refusal"]:
                n_ref += 1
                row["refusal_pass"] = ans_ok = metrics.refusal_ok(trace.answer)
                refusal_ok_n += int(ans_ok)
                real_fail = ans_ok is False
            else:
                n_ans += 1
                kw = metrics.keyword_hit(trace.answer, case["expected_keywords"])
                row["keyword_pass"] = ans_ok = kw
                kw_ok += int(kw)
                if judge_llm is not None:
                    from evals.judge import judge_answer
                    jr = judge_answer(judge_llm, case["question"],
                                      case.get("reference_answer", ""), trace.answer)
                    row["judge"] = jr
                    if jr.get("semantic_correct") is not None:
                        n_judged += 1
                        j_ok = bool(jr["semantic_correct"])
                        judge_ok += int(j_ok)
                        rel_sum += jr["relevancy"]
                        row["judge_pass"] = j_ok
                # 有效通过 = 规则过 或 judge 过（规则高精度低召回，judge 补假阴性）
                eff = kw or bool(row.get("judge_pass"))
                effective_ok += int(eff)
                # 真失败（写记忆）= 规则挂 且 judge 也挂；judge 认可的属于评测集误判，不入库
                real_fail = (kw is False) and (row.get("judge_pass") is False or judge_llm is None)
                if kw is False and row.get("judge_pass") is True:
                    row["eval_artifact"] = True
            if real_fail and memory is not None:
                outcome, root_cause, lesson = attribute_failure(llm, case, trace)
                memory.record(case["question"], outcome, root_cause, lesson)
                row["recorded_failure"] = {"outcome": outcome, "root_cause": root_cause[:80]}
            row["answer"] = (trace.answer or "")[:120]
            row["steps"] = trace.steps
            if trace.plan:
                row["plan"] = trace.plan
            steps_sum += trace.steps
            chars_sum += trace.prompt_chars + trace.completion_chars

        per_case.append(row)
        mark = "✓" if (hit or row.get("keyword_pass") or row.get("refusal_pass") or row.get("judge_pass")) else "✗"
        extra = ""
        if with_llm and not case["expect_refusal"]:
            extra += f" kw={row.get('keyword_pass')} judge={row.get('judge_pass', '-')}"
        print(f"  [{i}/{len(cases)}] {mark} {case['id']} hit@{top_k}={hit} rr={rr:.2f}{extra}", flush=True)

    aggregate = {
        "tag": tag,
        "mode": mode,
        "n_cases": len(per_case),
        "hit@5": round(hits / n_with_src, 4) if n_with_src else None,
        "mrr": round(mrr_sum / n_with_src, 4) if n_with_src else None,
        "recall@5": round(rec_sum / n_with_src, 4) if n_with_src else None,
        "ndcg@5": round(ndcg_sum / n_with_src, 4) if n_with_src else None,
        "keyword_acc": round(kw_ok / n_ans, 4) if n_ans else None,
        "judge_acc": round(judge_ok / n_judged, 4) if n_judged else None,
        "effective_acc": round(effective_ok / n_ans, 4) if (n_ans and judge_llm is not None) else None,
        "avg_relevancy": round(rel_sum / n_judged, 2) if n_judged else None,
        "refusal_acc": round(refusal_ok_n / n_ref, 4) if n_ref else None,
        "avg_steps": round(steps_sum / len(per_case), 2) if with_llm else None,
        "total_char_tokens": chars_sum if with_llm else None,
        "memory_on": memory_on,
        "elapsed_s": round(time.time() - started, 1),
    }

    config.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = config.HISTORY_DIR / f"eval-{tag}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for row in per_case:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps({"aggregate": aggregate}, ensure_ascii=False) + "\n")

    print("\n=== 聚合指标 ===")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")
    print(f"  历史文件: {out}")
    return {"aggregate": aggregate, "per_case": per_case}
