"""工具注册表：OpenAI function calling 格式的 schema + 执行器。

v1 只挂 retrieve 一个工具；新增工具在这里注册 schema 和分支即可。
"""
import json

from src import config
from src.context.compressor import format_evidence, pack_chunks
from src.trace import AgentTrace

TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "在中文新闻知识库中做混合检索（向量+BM25+RRF融合）。返回与查询最相关的若干文档段落。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询，应包含关键实体/事件名/数字词"},
                "k": {"type": "integer", "description": "返回段落数，默认 5，最大 8"},
            },
            "required": ["query"],
        },
    },
}]


def execute_tool(name: str, arguments: str, retriever, trace: AgentTrace,
                 compress: bool | None = None) -> str:
    """执行一次工具调用，返回给 LLM 的文本结果，并把调用写入 trace。"""
    if name != "retrieve":
        return f"Unknown tool: {name}"
    args = json.loads(arguments or "{}")
    query = args.get("query", "")
    k = min(int(args.get("k", config.TOP_K)), 8)
    chunks = retriever.retrieve(query, k=k)
    trace.tool_calls.append({"query": query, "k": k, "n_results": len(chunks)})
    trace.retrieved.extend(c.doc_id for c in chunks)
    if not chunks:
        return "(检索无结果)"

    use_compress = config.USE_COMPRESS if compress is None else compress
    if use_compress:
        chunks, stats = pack_chunks(chunks)
        prev = trace.compress or {"on": True, "before_chars": 0, "after_chars": 0, "dropped": 0}
        trace.compress = {
            "on": True,
            "before_chars": prev["before_chars"] + stats["before_chars"],
            "after_chars": prev["after_chars"] + stats["after_chars"],
            "dropped": prev["dropped"] + stats["dropped"],
            "budget": stats["budget"],
        }
    else:
        before = sum(len(c.parent_text or "") for c in chunks)
        prev = trace.compress or {"on": False, "before_chars": 0, "after_chars": 0, "dropped": 0}
        trace.compress = {
            "on": False,
            "before_chars": prev["before_chars"] + before,
            "after_chars": prev["after_chars"] + before,
            "dropped": prev["dropped"],
        }
    return format_evidence(chunks)