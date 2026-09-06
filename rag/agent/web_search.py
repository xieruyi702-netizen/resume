"""Tavily 网页检索：优先 API Key；无 Key 时走 keyless 试用。"""
from __future__ import annotations

import http.client
import json
import os
import ssl
from typing import Any

from llm import load_dotenv

load_dotenv()

TAVILY_HOST = "api.tavily.com"


def tavily_search(query: str, max_results: int = 5, *, include_answer: bool = False) -> dict[str, Any]:
    """调用 Tavily /search。返回标准化结果，失败时带 error 字段。"""
    query = (query or "").strip()
    if not query:
        return {"tool": "web_search", "error": "query 为空", "results": []}

    payload = {
        "query": query,
        "max_results": max(1, min(int(max_results or 5), 10)),
        "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", "basic"),
        "include_answer": include_answer,
        "include_raw_content": False,
    }
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    # 有 Key 用账户额度；否则 keyless 试用（有限额）
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        mode = "api_key"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
        mode = "keyless"

    body = json.dumps(payload).encode("utf-8")
    try:
        conn = http.client.HTTPSConnection(TAVILY_HOST, timeout=30, context=ssl.create_default_context())
        try:
            conn.request("POST", "/search", body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="ignore")
        finally:
            conn.close()
    except Exception as e:
        return {"tool": "web_search", "error": f"网络失败: {e}", "mode": mode, "results": []}

    if resp.status >= 400:
        return {
            "tool": "web_search",
            "error": f"HTTP {resp.status}: {raw[:240]}",
            "mode": mode,
            "results": [],
        }

    try:
        data = json.loads(raw)
    except Exception:
        return {"tool": "web_search", "error": f"响应非 JSON: {raw[:200]}", "mode": mode, "results": []}

    results = []
    for item in data.get("results") or []:
        results.append({
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "snippet": (item.get("content") or item.get("snippet") or "")[:400],
            "score": item.get("score"),
        })
    out: dict[str, Any] = {
        "tool": "web_search",
        "mode": mode,
        "query": query,
        "results": results,
    }
    if include_answer and data.get("answer"):
        out["answer"] = str(data["answer"])[:800]
    return out
