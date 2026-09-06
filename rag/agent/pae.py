#!/usr/bin/env python3
"""Plan-and-Execute 模式（对标 Claude Code Plan mode：先规划再执行）。

阶段：
  1) Plan  —— 产出 JSON 步骤表（LLM 或规则离线）
  2) Execute —— 按表顺序调 RecipeTools，解析 $from_search 绑定
  3) Synthesize —— 用证据合成最终回答（LLM 或模板）
"""
from __future__ import annotations

import json
import re
from typing import Any

from tools import RecipeTools

ALLOWED_TOOLS = {
    "keyword_search",
    "vector_search",
    "hybrid_search",
    "get_recipe",
    "get_section",
    "recall_memory",
    "web_search",
}

PLAN_SYSTEM = """你是美食汇 Plan 规划器（对标 Claude Code Plan mode）。
只输出一个 JSON 对象，不要 markdown，不要解释。格式：
{"goal":"一句话目标","steps":[{"tool":"工具名","args":{...},"why":"简短原因"}]}

可用工具：hybrid_search / keyword_search / vector_search / get_recipe / get_section / recall_memory / web_search
规则：
1) 本地优先：先 hybrid_search，再 get_recipe 或 get_section；有忌口/指代时加 recall_memory。
2) get_recipe/get_section 的 title 可写 "$from_search"（执行期绑定检索 Top1）。
3) 仅本地证据不足时才 web_search；步骤总数 ≤ 5。
4) 问原料/步骤时优先 get_section，否则 get_recipe。
"""

SYNTH_SYSTEM = """你是美食汇菜谱助手。根据【计划】与【工具证据】回答用户问题。
禁止编造证据中没有的步骤/用量/菜名；忌口必须遵守；证据不足则明确拒答。
简洁分点中文作答。
"""


def _guess_section(query: str) -> str | None:
    if re.search(r"(原料|食材|配料)", query):
        return "原料"
    if re.search(r"(步骤|怎么做|做法|教程)", query):
        return "步骤"
    if re.search(r"(用量|多少|几克)", query):
        return "用量"
    return None


def rule_plan(question: str, *, has_pre_hits: bool = True) -> dict:
    """无 LLM 时的确定性计划（可复现评测）。"""
    steps: list[dict] = [
        {"tool": "recall_memory", "args": {"query": question}, "why": "拉取忌口/会话"},
    ]
    steps.append({
        "tool": "hybrid_search",
        "args": {"query": question, "top_k": 5},
        "why": "确认/刷新命中白名单" if has_pre_hits else "本地检索",
        "optional_if_pre": bool(has_pre_hits),
    })
    sec = _guess_section(question)
    if sec:
        steps.append({
            "tool": "get_section",
            "args": {"title": "$from_search", "section": sec},
            "why": f"精读{sec}",
        })
    else:
        steps.append({
            "tool": "get_recipe",
            "args": {"title": "$from_search"},
            "why": "拉取全文",
        })
    return {"goal": f"回答：{question[:60]}", "steps": steps, "planner": "rule"}


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_plan(question: str, memory_block: str = "") -> dict:
    from llm import gen_client

    user = f"用户问题：{question}"
    if memory_block:
        user += f"\n\n【记忆摘要】\n{memory_block[:600]}"
    raw = gen_client().chat(
        [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=600,
    )
    data = _extract_json(raw) or {}
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list) or not steps:
        return rule_plan(question)
    clean = []
    for s in steps[:5]:
        if not isinstance(s, dict):
            continue
        tool = str(s.get("tool") or "").strip()
        if tool not in ALLOWED_TOOLS:
            continue
        args = s.get("args") if isinstance(s.get("args"), dict) else {}
        clean.append({"tool": tool, "args": args, "why": str(s.get("why") or "")[:80]})
    if not clean:
        return rule_plan(question)
    return {
        "goal": str(data.get("goal") or question)[:120],
        "steps": clean,
        "planner": "llm",
        "raw": raw[:400],
    }


def _bind_args(args: dict, *, focus_title: str | None) -> dict:
    out = dict(args or {})
    for k, v in list(out.items()):
        if v == "$from_search" and focus_title:
            out[k] = focus_title
    return out


def execute_plan(
    rt: RecipeTools,
    plan: dict,
    *,
    question: str,
    pre_hits: list[dict] | None = None,
) -> tuple[list[dict], str | None]:
    """执行计划，返回 trace 与焦点菜名。"""
    focus = None
    if pre_hits:
        focus = pre_hits[0].get("title")
        for h in pre_hits:
            if h.get("title"):
                rt.allowed_titles.add(h["title"])
    trace: list[dict] = [{"phase": "plan", "plan": {
        "goal": plan.get("goal"),
        "planner": plan.get("planner"),
        "steps": [
            {"tool": s.get("tool"), "args": s.get("args"), "why": s.get("why")}
            for s in (plan.get("steps") or [])
        ],
    }}]
    for i, step in enumerate(plan.get("steps") or []):
        tool = step.get("tool")
        args = _bind_args(step.get("args") or {}, focus_title=focus)
        if tool == "hybrid_search" and pre_hits and step.get("optional_if_pre"):
            result: dict[str, Any] = {
                "tool": "hybrid_search",
                "hits": pre_hits[:5],
                "skipped": "pre_hits",
            }
        else:
            result = rt.dispatch(str(tool), args)
        if isinstance(result, dict):
            hits = result.get("hits") or []
            if hits and isinstance(hits[0], dict) and hits[0].get("title"):
                focus = hits[0]["title"]
            if result.get("title") and not result.get("error"):
                focus = result.get("title") or focus
        trace.append({
            "phase": "execute",
            "step": i,
            "tool": tool,
            "arguments": args,
            "why": step.get("why"),
            "result": result,
        })
    return trace, focus


def synthesize(
    question: str,
    plan: dict,
    trace: list[dict],
    *,
    use_llm: bool,
    memory_block: str = "",
    focus_title: str | None = None,
) -> str:
    evidence_bits = []
    for t in trace:
        if t.get("phase") != "execute":
            continue
        r = t.get("result") or {}
        if not isinstance(r, dict):
            continue
        if r.get("content"):
            evidence_bits.append(f"【{r.get('title') or focus_title or '菜谱'}】\n{r['content'][:1200]}")
        elif r.get("text"):
            evidence_bits.append(f"【{r.get('section') or '节选'}·{r.get('title','')}】\n{r['text'][:800]}")
        elif r.get("hits"):
            titles = [h.get("title") for h in r["hits"][:3] if isinstance(h, dict)]
            evidence_bits.append("检索命中：" + "、".join(x for x in titles if x))
        elif r.get("error"):
            evidence_bits.append(f"工具错误({t.get('tool')}): {r['error']}")
    evidence = "\n\n".join(evidence_bits)[:3500]

    if not use_llm:
        for b in evidence_bits:
            if b.startswith("【") and "】\n" in b and not b.startswith("【工具"):
                return b
        return "证据不足，无法根据知识库回答该问题。"

    from llm import gen_client

    user = (
        f"用户问题：{question}\n\n"
        f"【计划】{json.dumps({'goal': plan.get('goal'), 'steps': plan.get('steps')}, ensure_ascii=False)}\n\n"
        f"【工具证据】\n{evidence or '（无）'}\n"
    )
    if memory_block:
        user = f"【记忆】\n{memory_block[:500]}\n\n" + user
    return gen_client().chat(
        [{"role": "system", "content": SYNTH_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=1000,
    )


def run_pae(
    question: str,
    rt: RecipeTools,
    *,
    memory_block: str = "",
    pre_hits: list[dict] | None = None,
    use_llm: bool = True,
) -> tuple[str, list[dict], dict]:
    """返回 (answer, trace, plan)。"""
    pre_hits = pre_hits or []
    if use_llm:
        try:
            plan = llm_plan(question, memory_block=memory_block)
        except Exception as e:
            plan = rule_plan(question, has_pre_hits=bool(pre_hits))
            plan["planner_error"] = str(e)[:120]
    else:
        plan = rule_plan(question, has_pre_hits=bool(pre_hits))

    trace, focus = execute_plan(rt, plan, question=question, pre_hits=pre_hits)
    answer = synthesize(
        question,
        plan,
        trace,
        use_llm=use_llm,
        memory_block=memory_block,
        focus_title=focus,
    )
    trace.append({"phase": "synthesize", "focus": focus, "answer_preview": (answer or "")[:200]})
    return answer, trace, plan
