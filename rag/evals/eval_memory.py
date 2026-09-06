#!/usr/bin/env python3
"""记忆遵从 / PreTool 门控轻量评测（默认无 LLM，可复现）。

覆盖：
1) 忌口写入 PROFILE + inject 可见
2) 会话滚动摘要在多轮后出现
3) get_recipe 菜名白名单拦截
4) web_search 本地空结果 streak 门控
5) offline agent 回合后忌口仍可注入
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from agentic import run_agentic  # noqa: E402
from memory import MemoryHub, SessionMemory  # noqa: E402
from tools import RecipeTools  # noqa: E402
from retriever import build_index  # noqa: E402


def _ok(name: str, cond: bool, detail: str = "") -> dict:
    return {"name": name, "pass": bool(cond), "detail": detail}


def test_allergy_profile(store: Path) -> dict:
    os.environ["MEISHI_MEMORY_STORE"] = str(store)
    hub = MemoryHub(user_id="eval_mem_allergy", persist_session=False)
    q = "我对花生过敏，以后别推荐含花生的菜"
    hub.after_turn(q, {"answer": "已记下忌口", "refused": False, "hits": []})
    profile = hub.memdir.profile_text()
    inject = hub.inject_block("推荐一道下饭菜")
    has_allergy = "花生" in profile or "过敏" in profile
    in_inject = "花生" in inject or "忌口" in inject or "过敏" in inject
    return _ok(
        "allergy_profile",
        has_allergy and in_inject,
        f"profile_has={has_allergy} inject_has={in_inject}",
    )


def test_session_summary() -> dict:
    s = SessionMemory(user_id="eval_sum", persist=False)
    for i in range(6):
        s.add("user", f"不吃香菜，想吃下饭菜第{i}问")
        s.add("assistant", f"好的第{i}答")
    s.set_focus(dish="番茄炒蛋")
    s.maybe_roll_summary()
    summary = s.working.get("session_summary", "")
    ctx = s.context_text()
    return _ok(
        "session_summary",
        bool(summary) and ("忌口" in summary or "香菜" in summary) and "会话摘要" in ctx,
        summary[:120],
    )


def test_title_whitelist(index) -> dict:
    rt = RecipeTools(index)
    hits = rt.hybrid_search("宫保鸡丁", top_k=3).get("hits") or []
    if not hits:
        return _ok("title_whitelist", False, "hybrid 无命中，无法测白名单")
    bad = rt.get_recipe(title="绝对不存在的虚构菜XYZ")
    blocked = bool(bad.get("error")) and "白名单" in str(bad.get("error"))
    title = hits[0]["title"]
    good = rt.get_recipe(title=title)
    allowed = not good.get("error") and good.get("title")
    return _ok("title_whitelist", blocked and bool(allowed), f"blocked={blocked} allowed={bool(allowed)}")


def test_web_empty_streak(index) -> dict:
    os.environ["MEISHI_WEB_EMPTY_STREAK"] = "2"
    rt = RecipeTools(index)
    rt._note_search_hits([])
    r1 = rt.web_search("随便")
    blocked1 = "拦截" in str(r1.get("error", ""))
    rt._note_search_hits([])
    r2 = rt.web_search("随便")
    streak_gate_cleared = "连续空结果未达" not in str(r2.get("error", ""))
    return _ok(
        "web_empty_streak",
        blocked1 and streak_gate_cleared,
        f"first_blocked={blocked1} second_cleared={streak_gate_cleared} err2={str(r2.get('error',''))[:80]}",
    )


def test_offline_agent_memory(index, store: Path) -> dict:
    os.environ["MEISHI_MEMORY_STORE"] = str(store)
    hub = MemoryHub(user_id="eval_agent_offline", persist_session=False)
    hub.after_turn("我不吃香菜", {"answer": "已记下", "refused": False, "hits": []})
    out = run_agentic("番茄炒蛋怎么做？", index=index, memory=hub, use_llm=False)
    inject = hub.inject_block("推荐")
    remembered = "香菜" in inject or "忌口" in inject
    answered = not out.get("refused")
    return _ok(
        "offline_agent_memory",
        remembered and answered,
        f"remembered={remembered} refused={out.get('refused')} mode={out.get('mode')}",
    )


def test_pae_offline(index) -> dict:
    hub = MemoryHub(user_id="eval_pae", persist_session=False)
    out = run_agentic("宫保鸡丁的原料有哪些", index=index, memory=hub, use_llm=False, mode="pae")
    plan = out.get("plan") or {}
    steps = plan.get("steps") or []
    has_plan = bool(steps) and plan.get("planner") == "rule"
    phases = {t.get("phase") for t in (out.get("trace") or []) if isinstance(t, dict) and t.get("phase")}
    has_phases = {"plan", "execute", "synthesize"} <= phases
    answered = not out.get("refused") and bool(out.get("answer"))
    # 原料题应走到 get_section
    tools_used = [t.get("tool") for t in (out.get("trace") or []) if t.get("phase") == "execute"]
    sectioned = "get_section" in tools_used
    return _ok(
        "pae_offline",
        has_plan and has_phases and answered and sectioned,
        f"planner={plan.get('planner')} phases={sorted(phases)} tools={tools_used} refused={out.get('refused')}",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    index = build_index()
    with tempfile.TemporaryDirectory() as td:
        store = Path(td) / "memory_store"
        store.mkdir(parents=True, exist_ok=True)
        results = [
            test_allergy_profile(store),
            test_session_summary(),
            test_title_whitelist(index),
            test_web_empty_streak(index),
            test_offline_agent_memory(index, store),
            test_pae_offline(index),
        ]

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    report = {
        "passed": passed,
        "total": total,
        "rate": round(passed / total, 4) if total else 0,
        "cases": results,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for r in results:
            mark = "PASS" if r["pass"] else "FAIL"
            print(f"[{mark}] {r['name']}: {r['detail']}")
        print(f"\nmemory eval: {passed}/{total} ({report['rate']})")
    out_path = Path(__file__).resolve().parent / "memory_metrics.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
