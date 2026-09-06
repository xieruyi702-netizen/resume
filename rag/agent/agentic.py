#!/usr/bin/env python3
"""Agentic RAG：LangChain ReAct 与 Plan-and-Execute 双模式。

- react：LangChain tool-calling Agent（边想边调工具）
- pae  ：Plan-and-Execute（先规划再执行再合成，对标 Claude Code Plan mode）
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from llm import load_dotenv
from memory import MemoryHub
from pae import run_pae
from retriever import build_index
from tools import TOOL_SPECS, RecipeTools

load_dotenv()

# 回退文案；正式以 AGENT.md（对标 CLAUDE.md）为准
_FALLBACK_SYSTEM = """你是「美食汇」Agentic RAG 助手。通过工具完成菜谱问答，禁止编造未检索到的步骤/用量。
工具：hybrid_search → get_recipe/get_section；忌口用 recall_memory；本地不足再 web_search。
模式：react（边想边调）或 pae（先计划后执行）。
"""


def load_agent_md() -> str:
    """加载项目说明书 AGENT.md。"""
    path = Path(__file__).resolve().parent / "AGENT.md"
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return _FALLBACK_SYSTEM


def _shrink(obj: Any, budget: int = 1200) -> Any:
    """工具结果预算（对标 Claude Code tool-result microcompact）。"""
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= budget:
        return obj
    if isinstance(obj, dict):
        slim = dict(obj)
        if "content" in slim and isinstance(slim["content"], str):
            slim["content"] = slim["content"][:500] + "…（已截断，可用 get_section 精读）"
        if "hits" in slim and isinstance(slim["hits"], list):
            slim["hits"] = [
                {
                    k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
                    for k, v in h.items()
                }
                if isinstance(h, dict)
                else h
                for h in slim["hits"][:3]
            ]
        if "text" in slim and isinstance(slim["text"], str):
            slim["text"] = slim["text"][:600] + "…"
        return slim
    return obj


def _dumps(obj: Any) -> str:
    return json.dumps(_shrink(obj), ensure_ascii=False)


def _build_langchain_tools(rt: RecipeTools, *, allow_web: bool = True):
    """把现有 RecipeTools 包装成 LangChain StructuredTool。"""
    from langchain_core.tools import StructuredTool

    def keyword_search(query: str, top_k: int = 5) -> str:
        """关键词检索（BM25/jieba）。适合菜名、原料等字面匹配强的问法。"""
        return _dumps(rt.keyword_search(query, top_k=top_k))

    def vector_search(query: str, top_k: int = 5) -> str:
        """向量语义检索（bge）。适合同义改写、模糊描述。"""
        return _dumps(rt.vector_search(query, top_k=top_k))

    def hybrid_search(query: str, top_k: int = 5) -> str:
        """关键词 + 向量双路 RRF 融合。不确定用哪路时优先调用。"""
        return _dumps(rt.hybrid_search(query, top_k=top_k))

    def get_recipe(title: str = "", doc_id: str = "") -> str:
        """按菜名或 doc_id 拉取整道菜谱正文（原料/用量/步骤）。"""
        return _dumps(rt.get_recipe(title=title or None, doc_id=doc_id or None))

    def get_section(title: str, section: str = "步骤") -> str:
        """读取某道菜的指定章节：原料 / 用量 / 步骤 / 简介 / 附加。"""
        return _dumps(rt.get_section(title, section))

    def recall_memory(query: str = "") -> str:
        """读取当前会话短期记忆与相关长期记忆（忌口/偏好/失败教训）。"""
        return _dumps(rt.recall_memory(query))

    def web_search(query: str, top_k: int = 5) -> str:
        """互联网检索（Tavily）。仅本地连续空结果后放行（RecipeTools 门控）。"""
        if not allow_web:
            return _dumps({"error": "本轮关闭 web_search"})
        return _dumps(rt.web_search(query, top_k=top_k))

    return [
        StructuredTool.from_function(keyword_search),
        StructuredTool.from_function(vector_search),
        StructuredTool.from_function(hybrid_search),
        StructuredTool.from_function(get_recipe),
        StructuredTool.from_function(get_section),
        StructuredTool.from_function(recall_memory),
        StructuredTool.from_function(web_search),
    ]


def _chat_model():
    """OpenAI 兼容 Chat（智谱 / DeepSeek 等，走 ZHIPU_BASE_URL）。"""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("MEISHI_LLM_KEY") or ""
    base_url = (os.environ.get("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
    model = os.environ.get("MEISHI_GEN_MODEL", "deepseek-v4-flash")
    if not api_key:
        raise RuntimeError("缺少 ZHIPU_API_KEY")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_tokens=1200,
    )


def _run_langchain_agent(
    question: str,
    rt: RecipeTools,
    *,
    max_steps: int = 4,
    memory_block: str = "",
    allow_web: bool = True,
) -> tuple[str, list[dict]]:
    """LangChain tool-calling Agent + AgentExecutor（不依赖 LangGraph）。"""
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    tools = _build_langchain_tools(rt, allow_web=allow_web)
    llm = _chat_model()
    system = load_agent_md()
    if memory_block:
        system = (
            system
            + "\n\n【记忆上下文（Memdir/PROFILE → 近轮会话）】\n"
            + memory_block
            + "\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=max(1, max_steps),
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    result = executor.invoke({"input": f"用户问题：{question}"})
    final = str(result.get("output") or "").strip()
    trace: list[dict] = []
    for i, step in enumerate(result.get("intermediate_steps") or []):
        action, observation = step[0], step[1]
        entry: dict[str, Any] = {
            "step": i,
            "tool": getattr(action, "tool", None),
            "arguments": getattr(action, "tool_input", {}) or {},
        }
        try:
            entry["result"] = json.loads(observation) if isinstance(observation, str) else observation
        except Exception:
            entry["result"] = {"raw": str(observation)[:500]}
        trace.append(entry)

    if final:
        trace.append({"step": len(trace), "final": final[:200]})
    else:
        search = rt.hybrid_search(question, top_k=5)
        hits = search.get("hits") or []
        title = hits[0]["title"] if hits else None
        recipe = rt.get_recipe(title=title) if title else None
        if recipe and not recipe.get("error"):
            final = f"根据检索，推荐【{recipe['title']}】：\n{(recipe.get('content') or '')[:1000]}"
        else:
            final = "证据不足，无法根据知识库回答该问题。"
        trace.append({"step": len(trace), "fallback": True, "final": final[:200]})

    return final, trace


def run_agentic(
    question: str,
    *,
    index=None,
    memory: MemoryHub | None = None,
    max_steps: int = 4,
    use_llm: bool = True,
    user_id: str = "default",
    mode: str = "react",
) -> dict:
    """mode: react | pae（Plan-and-Execute）。"""
    from agent import should_refuse

    mode = (mode or "react").strip().lower()
    if mode in {"plan", "plan-and-execute", "plan_and_execute"}:
        mode = "pae"

    index = index or build_index()
    memory = memory or MemoryHub(user_id=user_id)
    tools = RecipeTools(index, memory=memory)
    q = memory.session.rewrite_query(question)

    # PreTool 硬门控：先本地 hybrid，OOD 直接拒答（对齐非 agentic 路径）
    pre = tools.hybrid_search(q, top_k=5)
    pre_hits = pre.get("hits") or []
    hit_tuples = [
        (
            {
                "title": h.get("title", ""),
                "doc_id": h.get("doc_id", ""),
                "text": h.get("snippet") or h.get("text") or "",
                "section": h.get("section", ""),
                "category_zh": h.get("category_zh", ""),
            },
            float(h.get("score") or 0),
        )
        for h in pre_hits
    ]
    if should_refuse(q, hit_tuples):
        out = {
            "query": question,
            "resolved_query": q,
            "mode": f"{mode}-ood-gate",
            "agent_mode": mode,
            "trace": [{"tool": "hybrid_search", "result": {"hits": pre_hits[:3]}, "gate": "should_refuse"}],
            "answer": "该问题超出美食汇菜谱知识库范围，或证据不足，已拒绝编造。",
            "refused": True,
            "memory_used": False,
        }
        memory.after_turn(question, out)
        return out

    mem_block = memory.inject_block(q)

    # ---------- Plan-and-Execute ----------
    if mode == "pae":
        final_answer, trace, plan = run_pae(
            q,
            tools,
            memory_block=mem_block,
            pre_hits=pre_hits,
            use_llm=use_llm,
        )
        out = {
            "query": question,
            "resolved_query": q,
            "mode": "agentic-pae" if use_llm else "agentic-pae-offline",
            "agent_mode": "pae",
            "plan": {
                "goal": plan.get("goal"),
                "planner": plan.get("planner"),
                "steps": plan.get("steps"),
            },
            "trace": [{"tool": "hybrid_search", "result": {"hits": pre_hits[:3]}, "gate": "pre"}] + trace,
            "answer": final_answer,
            "refused": "无法" in final_answer or "证据不足" in final_answer or "拒绝" in final_answer,
            "tools": [t["function"]["name"] for t in TOOL_SPECS],
            "memory_used": bool(mem_block),
            "allowed_titles": sorted(tools.allowed_titles)[:8],
            "empty_search_streak": tools.empty_search_streak,
        }
        memory.after_turn(question, out)
        return out

    # ---------- ReAct（LangChain AgentExecutor）----------
    if not use_llm:
        mem = tools.recall_memory(q)
        title = pre_hits[0]["title"] if pre_hits else None
        recipe = tools.get_recipe(title=title) if title else {"error": "no hit"}
        if recipe.get("error"):
            ans = "知识库未找到相关菜谱，已拒绝编造。"
            refused = True
        else:
            ans = f"【{recipe['title']}】\n{(recipe.get('content') or '')[:1200]}"
            refused = False
        result = {
            "query": question,
            "resolved_query": q,
            "mode": "agentic-offline",
            "agent_mode": "react",
            "trace": [
                {"tool": "recall_memory", "result": mem},
                {"tool": "hybrid_search", "result": {"hits": pre_hits[:3]}},
                {"tool": "get_recipe", "result": {"title": recipe.get("title"), "error": recipe.get("error")}},
            ],
            "answer": ans,
            "refused": refused,
        }
        memory.after_turn(question, result)
        return result

    final_answer, trace = _run_langchain_agent(
        q, tools, max_steps=max_steps, memory_block=mem_block, allow_web=True
    )
    out = {
        "query": question,
        "resolved_query": q,
        "mode": "agentic-langchain",
        "agent_mode": "react",
        "framework": "langchain",
        "trace": [{"tool": "hybrid_search", "result": {"hits": pre_hits[:3]}, "gate": "pre"}] + trace,
        "answer": final_answer,
        "refused": "无法" in final_answer or "证据不足" in final_answer or "拒绝" in final_answer,
        "tools": [t["function"]["name"] for t in TOOL_SPECS],
        "memory_used": bool(mem_block),
        "allowed_titles": sorted(tools.allowed_titles)[:8],
        "empty_search_streak": tools.empty_search_streak,
    }
    memory.after_turn(question, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="美食汇 Agentic RAG（ReAct / Plan-and-Execute）")
    ap.add_argument("query", nargs="?", default="宫保鸡丁怎么做？")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--mode", choices=["react", "pae"], default=os.environ.get("MEISHI_AGENT_MODE", "react"))
    ap.add_argument("--user", default="default", help="用户 id，记忆按用户隔离")
    ap.add_argument("--session", default=None, help="同一用户下的会话 id")
    args = ap.parse_args()
    mem = MemoryHub(user_id=args.user, session_id=args.session)
    print(json.dumps(
        run_agentic(
            args.query,
            memory=mem,
            max_steps=args.max_steps,
            use_llm=not args.no_llm,
            mode=args.mode,
        ),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
