#!/usr/bin/env python3
"""Agentic RAG：基于 LangChain / LangGraph ReAct Agent（工具调用循环）。"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from llm import load_dotenv
from memory import MemoryHub
from retriever import build_index
from tools import TOOL_SPECS, RecipeTools

load_dotenv()

AGENT_SYSTEM = """你是「美食汇」Agentic RAG 助手。通过工具完成菜谱问答，禁止编造未检索到的步骤/用量。

工具策略：
1) 先 hybrid_search（或 keyword_search / vector_search）；
2) 锁定菜名后 get_recipe 或 get_section 精读；
3) 有指代/忌口时 recall_memory；
4) 本地证据不足再 web_search，并注明网络来源；
5) 证据足够后直接给出面向用户的中文最终答案。
"""


def _shrink(obj: Any) -> Any:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= 1800:
        return obj
    if isinstance(obj, dict):
        slim = dict(obj)
        if "content" in slim and isinstance(slim["content"], str):
            slim["content"] = slim["content"][:800] + "…"
        if "hits" in slim and isinstance(slim["hits"], list):
            slim["hits"] = slim["hits"][:3]
        return slim
    return obj


def _dumps(obj: Any) -> str:
    return json.dumps(_shrink(obj), ensure_ascii=False)


def _build_langchain_tools(rt: RecipeTools):
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
        """互联网检索（Tavily）。仅本地知识库不足时使用。"""
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
) -> tuple[str, list[dict]]:
    """LangGraph create_react_agent：模型 tool-calling → 执行工具 → 直到最终回答。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    tools = _build_langchain_tools(rt)
    llm = _chat_model()
    # 上下文组装：系统策略 → 预算内记忆（长期约束优先）→ 当前问题；检索证据走工具观察
    prompt = AGENT_SYSTEM
    if memory_block:
        prompt = (
            AGENT_SYSTEM
            + "\n\n【记忆上下文（已按预算组装：长期约束 → 近轮会话）】\n"
            + memory_block
            + "\n"
        )
    agent = create_react_agent(llm, tools, prompt=prompt)

    # recursion_limit ≈ 每轮模型+工具各算一步，留余量
    result = agent.invoke(
        {"messages": [HumanMessage(content=f"用户问题：{question}")]},
        config={"recursion_limit": max(10, max_steps * 3)},
    )
    messages = result.get("messages") or []
    trace: list[dict] = []
    final = ""

    for i, m in enumerate(messages):
        if isinstance(m, AIMessage):
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    trace.append({
                        "step": len(trace),
                        "tool": tc.get("name"),
                        "arguments": tc.get("args") or {},
                    })
            content = (m.content or "").strip() if isinstance(m.content, str) else str(m.content or "")
            if content and not tool_calls:
                final = content
                trace.append({"step": len(trace), "final": final[:200]})
        elif isinstance(m, ToolMessage):
            # 把工具观察挂到最近一条 tool trace
            for t in reversed(trace):
                if t.get("tool") and "result" not in t:
                    try:
                        t["result"] = json.loads(m.content) if isinstance(m.content, str) else m.content
                    except Exception:
                        t["result"] = {"raw": str(m.content)[:500]}
                    break

    if not final:
        # 兜底：强制 hybrid + get_recipe
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
) -> dict:
    index = index or build_index()
    memory = memory or MemoryHub(user_id=user_id)
    tools = RecipeTools(index, memory=memory)
    q = memory.session.rewrite_query(question)

    # 无 LLM：确定性工具路径（hybrid → get_recipe → 模板答案）
    if not use_llm:
        mem = tools.recall_memory(q)
        search = tools.hybrid_search(q, top_k=5)
        hits = search.get("hits") or []
        title = hits[0]["title"] if hits else None
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
            "trace": [
                {"tool": "recall_memory", "result": mem},
                {"tool": "hybrid_search", "result": {"hits": hits[:3]}},
                {"tool": "get_recipe", "result": {"title": recipe.get("title"), "error": recipe.get("error")}},
            ],
            "answer": ans,
            "refused": refused,
        }
        memory.after_turn(question, result)
        return result

    mem_block = memory.inject_block(q)
    final_answer, trace = _run_langchain_agent(
        q, tools, max_steps=max_steps, memory_block=mem_block
    )
    out = {
        "query": question,
        "resolved_query": q,
        "mode": "agentic-langchain",
        "framework": "langchain+langgraph",
        "trace": trace,
        "answer": final_answer,
        "refused": "无法" in final_answer or "证据不足" in final_answer or "拒绝" in final_answer,
        "tools": [t["function"]["name"] for t in TOOL_SPECS],
        "memory_used": bool(mem_block),
    }
    memory.after_turn(question, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="美食汇 Agentic RAG（LangChain）")
    ap.add_argument("query", nargs="?", default="宫保鸡丁怎么做？")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--user", default="default", help="用户 id，记忆按用户隔离")
    ap.add_argument("--session", default=None, help="同一用户下的会话 id")
    args = ap.parse_args()
    mem = MemoryHub(user_id=args.user, session_id=args.session)
    print(json.dumps(
        run_agentic(args.query, memory=mem, max_steps=args.max_steps, use_llm=not args.no_llm),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
