"""美食汇 Agentic RAG 工具集：关键词/向量检索 / 读菜谱 / 记忆 / 网页检索。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from retriever import HybridIndex, load_chunks
from web_search import tavily_search


TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "关键词检索（BM25/jieba）。适合菜名、原料等字面匹配强的问法。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问句"},
                    "top_k": {"type": "integer", "description": "返回条数，默认5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "向量语义检索（bge-small-zh）。适合同义改写、模糊描述、语义相近菜名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问句"},
                    "top_k": {"type": "integer", "description": "返回条数，默认5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": "关键词 + 向量双路检索后 RRF 融合。不确定用哪路时优先调用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问句"},
                    "top_k": {"type": "integer", "description": "返回条数，默认5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe",
            "description": "按菜名或 doc_id 拉取整道菜谱正文（含原料/用量/步骤），用于精读一篇菜。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "菜名，如 宫保鸡丁"},
                    "doc_id": {"type": "string", "description": "文档 id，如 meat_dish/宫保鸡丁/宫保鸡丁"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": "读取某道菜的指定章节：原料 / 用量 / 步骤 / 简介。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "菜名"},
                    "section": {
                        "type": "string",
                        "enum": ["原料", "用量", "步骤", "简介", "附加"],
                        "description": "章节名",
                    },
                },
                "required": ["title", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "读取当前会话短期记忆与相关长期记忆（忌口/偏好/失败教训）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用于召回长期记忆的查询"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "互联网检索（Tavily）。仅当本地知识库不足、需要补充时令/替代做法/外部资料时使用；烹饪主答案仍优先本地菜谱。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "网页搜索问句"},
                    "top_k": {"type": "integer", "description": "返回条数，默认5"},
                },
                "required": ["query"],
            },
        },
    },
]


class RecipeTools:
    def __init__(self, index: HybridIndex, memory=None, recipes_path: Path | None = None):
        self.index = index
        self.memory = memory
        root = Path(__file__).resolve().parents[1]
        self.recipes_path = recipes_path or (root / "kb" / "recipes.jsonl")
        self._recipes: list[dict] | None = None
        self._by_title: dict[str, dict] = {}
        self._by_id: dict[str, dict] = {}
        # PreTool 状态：本轮检索命中菜名白名单；本地连续空结果次数
        self.allowed_titles: set[str] = set()
        self.empty_search_streak = 0

    def _note_search_hits(self, hits: list[dict]) -> None:
        titles = {h.get("title") for h in hits if isinstance(h, dict) and h.get("title")}
        self.allowed_titles |= {t for t in titles if t}
        if hits:
            self.empty_search_streak = 0
        else:
            self.empty_search_streak += 1

    def _title_allowed(self, title: str | None) -> bool:
        if not title:
            return True  # 仅 doc_id 时由下方解析
        if not self.allowed_titles:
            return True  # 尚未检索过则放行（offline 路径）
        if title in self.allowed_titles:
            return True
        return any(title in t or t in title for t in self.allowed_titles)

    def _ensure_recipes(self) -> None:
        if self._recipes is not None:
            return
        rows = []
        if self.recipes_path.exists():
            with self.recipes_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        self._recipes = rows
        for r in rows:
            self._by_id[r["id"]] = r
            self._by_title[r["title"]] = r

    def _fmt_hits(self, hits: list[tuple[dict, float]]) -> list[dict]:
        out = []
        for c, s in hits:
            out.append({
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "title": c.get("title"),
                "section": c.get("section"),
                "score": round(float(s), 4),
                "snippet": (c.get("text") or "")[:280],
            })
        return out

    def keyword_search(self, query: str, top_k: int = 5) -> dict:
        hits = self.index.search(query, top_k=top_k, mode="bm25")
        fmt = self._fmt_hits(hits)
        self._note_search_hits(fmt)
        return {"tool": "keyword_search", "hits": fmt}

    def vector_search(self, query: str, top_k: int = 5) -> dict:
        if self.index.vector is None:
            self._note_search_hits([])
            return {"tool": "vector_search", "error": "向量索引未加载", "hits": []}
        hits = self.index.search(query, top_k=top_k, mode="dense")
        fmt = self._fmt_hits(hits)
        self._note_search_hits(fmt)
        return {"tool": "vector_search", "hits": fmt}

    def hybrid_search(self, query: str, top_k: int = 5) -> dict:
        hits = self.index.search(query, top_k=top_k, mode="hybrid")
        fmt = self._fmt_hits(hits)
        self._note_search_hits(fmt)
        return {"tool": "hybrid_search", "hits": fmt}

    def get_recipe(self, title: str | None = None, doc_id: str | None = None) -> dict:
        if title and not self._title_allowed(title):
            return {
                "tool": "get_recipe",
                "error": "菜名不在本轮检索命中白名单，请先 hybrid_search",
                "title": title,
                "allowed": sorted(self.allowed_titles)[:10],
            }
        self._ensure_recipes()
        doc = None
        if doc_id and doc_id in self._by_id:
            doc = self._by_id[doc_id]
        elif title:
            doc = self._by_title.get(title)
            if doc is None:
                for t, d in self._by_title.items():
                    if title in t or t in title:
                        doc = d
                        break
        if not doc:
            return {"tool": "get_recipe", "error": "未找到菜谱", "title": title, "doc_id": doc_id}
        # doc_id 路径也校验标题白名单（若已有检索）
        if self.allowed_titles and doc["title"] not in self.allowed_titles:
            if not any(doc["title"] in t or t in doc["title"] for t in self.allowed_titles):
                return {
                    "tool": "get_recipe",
                    "error": "菜名不在本轮检索命中白名单，请先 hybrid_search",
                    "title": doc["title"],
                }
        return {
            "tool": "get_recipe",
            "doc_id": doc["id"],
            "title": doc["title"],
            "category_zh": doc.get("category_zh"),
            "content": (doc.get("content") or "")[:2500],
        }

    def get_section(self, title: str, section: str) -> dict:
        recipe = self.get_recipe(title=title)
        if recipe.get("error"):
            return {**recipe, "tool": "get_section"}
        text = recipe.get("content") or ""
        markers = {
            "简介": "简介：",
            "原料": "原料：",
            "用量": "用量计算：",
            "步骤": "步骤：",
            "附加": "附加：",
        }
        key = markers.get(section, section)
        if key not in text:
            return {"tool": "get_section", "title": title, "section": section, "text": ""}
        start = text.find(key) + len(key)
        rest = text[start:]
        next_pos = len(rest)
        for m in markers.values():
            if m == key:
                continue
            p = rest.find("\n\n" + m)
            if p >= 0:
                next_pos = min(next_pos, p)
        body = rest[:next_pos].strip()[:1200]
        return {"tool": "get_section", "title": recipe["title"], "section": section, "text": body}

    def recall_memory(self, query: str = "") -> dict:
        if not self.memory:
            return {"tool": "recall_memory", "text": ""}
        return {"tool": "recall_memory", "text": self.memory.inject_block(query or "")}

    def web_search(self, query: str, top_k: int = 5) -> dict:
        need = int(os.environ.get("MEISHI_WEB_EMPTY_STREAK", "2"))
        if self.empty_search_streak < need:
            return {
                "tool": "web_search",
                "error": f"本地检索连续空结果未达 {need} 次，已拦截 web_search",
                "empty_streak": self.empty_search_streak,
            }
        return tavily_search(query, max_results=top_k)

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict:
        args = arguments or {}
        top_k = int(args.get("top_k") or 5)
        if name == "keyword_search":
            return self.keyword_search(args.get("query", ""), top_k=top_k)
        if name == "vector_search":
            return self.vector_search(args.get("query", ""), top_k=top_k)
        if name == "hybrid_search":
            return self.hybrid_search(args.get("query", ""), top_k=top_k)
        if name == "get_recipe":
            return self.get_recipe(title=args.get("title"), doc_id=args.get("doc_id"))
        if name == "get_section":
            return self.get_section(args.get("title", ""), args.get("section", "步骤"))
        if name == "recall_memory":
            return self.recall_memory(args.get("query", ""))
        if name == "web_search":
            return self.web_search(args.get("query", ""), top_k=top_k)
        return {"error": f"unknown tool: {name}"}
