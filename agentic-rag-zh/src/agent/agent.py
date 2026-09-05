"""ReAct 风格 agent：LLM ↔ retrieve 工具循环，支持失败记忆注入。

mode 说明：
  memory_on=True  → 回答前从失败记忆库检索相似失败案例，注入 system prompt
  memory_on=False → 不注入（对照组），失败仍会被写入记忆库供后续运行使用
"""
from src import config
from src.agent.tools import TOOLS_SCHEMA, execute_tool
from src.llm.provider import get_llm
from src.retrieval.retriever import get_retriever
from src.trace import AgentTrace

SYSTEM_PROMPT = """你是一个严谨的中文知识库问答助手。你有一个工具 retrieve(query, k)，用于在新闻知识库中做混合检索。

规则：
1. 回答任何问题前必须先调用 retrieve 检索，禁止凭记忆作答。
2. 把问题改写成检索查询（提取实体、事件名、数字词）；若结果不好，换一个表述再检索（最多再试一次）。
3. 只依据检索到的内容回答，答案末尾标注来源，如（来源: doc_xxx）。
4. 若检索内容不包含答案、或问题超出知识库范围，必须如实回答："知识库中没有找到相关信息，无法回答。"禁止编造。"""


class Agent:
    def __init__(self, llm=None, retriever=None, memory=None):
        self.llm = llm or get_llm()
        self.retriever = retriever or get_retriever()
        self.memory = memory

    def run(self, question: str, memory_on: bool = True) -> AgentTrace:
        trace = AgentTrace()
        system = SYSTEM_PROMPT
        if self.memory and memory_on:
            hits = self.memory.recall(question)
            if hits:
                lines = "\n".join(
                    f"- 相似问题「{h['question']}」曾失败({h['outcome']})，根因: {h['root_cause']}；教训: {h['lesson']}"
                    for h in hits
                )
                system += "\n\n[历史失败案例提醒] 以下是过去相似任务失败后总结的教训，回答时注意规避：\n" + lines

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
        try:
            for step in range(config.MAX_AGENT_STEPS):
                trace.steps = step + 1
                trace.prompt_chars += sum(len(m.get("content") or "") for m in messages)
                resp = self.llm.chat(messages, tools=TOOLS_SCHEMA)
                trace.completion_chars += len(resp.content or "")
                if resp.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [
                            {"id": t["id"], "type": "function",
                             "function": {"name": t["name"], "arguments": t["arguments"]}}
                            for t in resp.tool_calls
                        ],
                    })
                    for t in resp.tool_calls:
                        result = execute_tool(t["name"], t["arguments"], self.retriever, trace)
                        messages.append({"role": "tool", "tool_call_id": t["id"], "content": result})
                else:
                    trace.answer = resp.content or ""
                    break
            if not trace.answer:
                trace.answer = "(agent 达到最大步数，未产出最终答案)"
        except Exception as e:
            trace.error = str(e)
            trace.answer = trace.answer or f"(agent 出错: {e})"
        return trace
