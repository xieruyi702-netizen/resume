"""Plan-and-Execute agent：规划 → 执行 → 失败重规划 → 汇总作答。

与 ReAct（graph_react 中的 Agent）同接口：run(question, memory_on) -> AgentTrace，
评测 runner 可通过 --mode react|pae 切换，其余（检索器/Judge/记忆）完全复用。

流程：
  1. planner    一次输出结构化 JSON 检索计划（1~3 步；含失败记忆提醒）
  2. executor   逐步执行 retrieve，跨步骤按 parent 去重汇总证据
  3. replanner  某步检索为空时，带上失败上下文重写该步 query（每步至多重试 1 次）
  4. answer     汇总全部证据一次性作答（引用/拒答规则与 ReAct 相同）
"""
import json
import re

from src import config
from src.trace import AgentTrace

PLANNER_SYSTEM = """你是 RAG 系统的检索规划器。针对用户问题制定检索计划，只输出一个 JSON 对象：
{"understanding": "一句话问题理解",
 "steps": [{"id": 1, "query": "检索词", "expect": "该步要确认的事实点"}],
 "answer_strategy": "如何用检索结果组织回答"}
要求：
- 检索词必须使用中文（知识库为中文语料，禁止输出英文检索词）
- 1~3 步：简单事实问题 1 步即可，不要过度规划；跨文档、对比、多要点问题才拆多步
- query 必须包含具体实体/数字/事件名，不要照抄整个问题
- 每步 expect 说明该步要确认什么，供执行后校验"""

REPLANNER_PROMPT = """检索步骤失败。原查询「{query}」没有命中任何结果。
问题：{question}
请换一个表述重写检索词（更具体的实体/同义词/事件名），只输出 JSON：
{{"query": "新检索词"}}"""

ANSWER_SYSTEM = """你是一个严谨的中文知识库问答助手。下面已按检索计划收集好资料，请回答用户问题。
规则：
1. 只依据资料内容回答，答案末尾标注来源，如（来源: doc_xxx）。
2. 若资料与问题明显无关、不包含答案、或问题超出知识库范围，必须如实回答："知识库中没有找到相关信息，无法回答。"禁止用不相关的资料拼凑回答。
3. 按问题的各个要点组织回答，不要遗漏计划中要求确认的事实点。"""


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class PlanExecuteAgent:
    def __init__(self, llm=None, retriever=None, memory=None):
        from src.llm.provider import get_llm
        from src.retrieval.retriever import get_retriever
        self.llm = llm or get_llm()
        self.retriever = retriever or get_retriever()
        self.memory = memory

    # ---- planner ----
    def _plan(self, question: str, memory_hits: list[dict], trace: AgentTrace) -> dict:
        system = PLANNER_SYSTEM
        if memory_hits:
            lines = "\n".join(
                f"- 相似问题「{h['question']}」曾失败({h['outcome']})，根因: {h['root_cause']}；教训: {h['lesson']}"
                for h in memory_hits
            )
            system += "\n\n[历史失败案例提醒] 制定检索计划时注意规避：\n" + lines
        resp = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"用户问题：{question}"},
        ])
        trace.steps += 1
        trace.prompt_chars += len(system) + len(question)
        trace.completion_chars += len(resp.content or "")
        plan = _parse_json(resp.content or "")
        if not plan or not plan.get("steps"):
            # 兜底：解析失败退化为单步计划（等价于直接检索）
            plan = {"understanding": "直接检索", "answer_strategy": "依据检索结果回答",
                    "steps": [{"id": 1, "query": question, "expect": "相关文档"}],
                    "fallback": True}
        plan["steps"] = [s for s in plan["steps"] if s.get("query")][: config.MAX_PLAN_STEPS]
        if not plan["steps"]:
            plan["steps"] = [{"id": 1, "query": question, "expect": "相关文档"}]
        return plan

    # ---- executor + replanner ----
    def _execute(self, question: str, plan: dict, trace: AgentTrace) -> list:
        """执行各步检索，返回去重后的证据块列表。"""
        evidence, seen_parents = [], set()
        for step in plan["steps"]:
            query = step["query"]
            chunks = self.retriever.retrieve(query, k=config.TOP_K)
            trace.tool_calls.append({"query": query, "k": config.TOP_K, "n_results": len(chunks)})
            trace.retrieved.extend(c.doc_id for c in chunks)

            if not chunks:  # 重规划：换表述重试一次
                resp = self.llm.chat([{"role": "user", "content": REPLANNER_PROMPT.format(
                    query=query, question=question)}])
                trace.steps += 1
                trace.completion_chars += len(resp.content or "")
                new_q = (_parse_json(resp.content or "") or {}).get("query")
                if new_q:
                    chunks = self.retriever.retrieve(new_q, k=config.TOP_K)
                    trace.tool_calls.append({"query": new_q, "k": config.TOP_K,
                                             "n_results": len(chunks), "replan": True})
                    trace.retrieved.extend(c.doc_id for c in chunks)

            for c in chunks:  # 跨步骤 parent 去重
                if c.parent_id in seen_parents:
                    continue
                seen_parents.add(c.parent_id)
                evidence.append(c)
                if len(evidence) >= 8:  # 证据总量上限，控制回答阶段上下文
                    return evidence
        return evidence

    # ---- answer ----
    def _answer(self, question: str, plan: dict, evidence: list, trace: AgentTrace) -> str:
        if evidence:
            material = "\n\n".join(
                f"[{i}] (来源: {c.doc_id}) {c.parent_text}" for i, c in enumerate(evidence, 1)
            )
            user = f"用户问题：{question}\n\n检索到的资料：\n{material}\n\n请回答。"
        else:
            user = f"用户问题：{question}\n\n（检索无任何结果）"
        resp = self.llm.chat([{"role": "system", "content": ANSWER_SYSTEM},
                              {"role": "user", "content": user}])
        trace.steps += 1
        trace.prompt_chars += len(user) + len(ANSWER_SYSTEM)
        trace.completion_chars += len(resp.content or "")
        return resp.content or ""

    def run(self, question: str, memory_on: bool = True) -> AgentTrace:
        trace = AgentTrace()
        memory_hits = self.memory.recall(question) if (self.memory and memory_on) else []
        try:
            plan = self._plan(question, memory_hits, trace)
            trace.plan = {"understanding": plan.get("understanding", ""),
                          "n_steps": len(plan["steps"]),
                          "queries": [s["query"][:40] for s in plan["steps"]],
                          "fallback": plan.get("fallback", False)}
            evidence = self._execute(question, plan, trace)
            trace.answer = self._answer(question, plan, evidence, trace)
        except Exception as e:
            trace.error = str(e)
            trace.answer = trace.answer or f"(agent 出错: {e})"
        return trace
