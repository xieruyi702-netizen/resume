"""长期失败记忆：LLM 根因归因 → 向量库存 → 新任务检索相似失败注入 prompt。

防污染三道闸：
  1. 相似度阈值（MEMORY_MIN_SIM）卡入口，不相似不注入；
  2. 只存 lesson/根因摘要，不存错误答案全文；
  3. 注入上限 MEMORY_TOP_K 条。
"""
import json
import re

import chromadb

from src import config

ATTRIBUTOR_PROMPT = """你是 RAG 系统质量分析专家。下面是一次失败的问答案例（问题、检索命中、回答）。只输出一个 JSON 对象，不要输出其他内容：
{{"outcome": "wrong_answer|missed_refusal|tool_misuse|no_retrieval", "root_cause": "一句话根因", "lesson": "一句话可执行教训（下次检索/回答该怎么改进）"}}

问题: {q}
检索命中文档: {docs}
回答: {a}
参考答案要点: {ref}"""


class FailureMemory:
    def __init__(self):
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.col = client.get_or_create_collection(
            config.MEMORY_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def record(self, question: str, outcome: str, root_cause: str, lesson: str) -> None:
        from src.retrieval.embedder import encode
        self.col.add(
            ids=[f"f{self.col.count() + 1}-{abs(hash(question)) % 100000}"],
            documents=[question],
            embeddings=encode([question]),
            metadatas=[{
                "outcome": outcome,
                "root_cause": (root_cause or "")[:300],
                "lesson": (lesson or "")[:300],
            }],
        )

    def recall(self, question: str, top_k: int | None = None,
               min_sim: float | None = None) -> list[dict]:
        from src.retrieval.embedder import encode
        top_k = top_k or config.MEMORY_TOP_K
        min_sim = config.MEMORY_MIN_SIM if min_sim is None else min_sim
        if self.col.count() == 0:
            return []
        res = self.col.query(query_embeddings=encode([question]),
                             n_results=min(top_k, self.col.count()))
        hits = []
        for i, cid in enumerate(res["ids"][0]):
            sim = 1.0 - res["distances"][0][i]
            if sim >= min_sim:
                meta = res["metadatas"][0][i]
                hits.append({"question": res["documents"][0][i], **meta, "similarity": round(sim, 3)})
        return hits


def attribute_failure(llm, case: dict, trace) -> tuple[str, str, str]:
    """LLM 根因归因；解析失败时给保守兜底，保证记忆链路不断。"""
    prompt = ATTRIBUTOR_PROMPT.format(
        q=case["question"],
        docs=",".join(dict.fromkeys(trace.retrieved))[:400] or "(无)",
        a=(trace.answer or "")[:500],
        ref="; ".join(case.get("expected_keywords") or []) or case.get("reference_answer", "")[:200],
    )
    resp = llm.chat([{"role": "user", "content": prompt}])
    try:
        m = re.search(r"\{.*\}", resp.content or "", re.S)
        obj = json.loads(m.group(0))
        return (
            obj.get("outcome", "wrong_answer"),
            obj.get("root_cause", "未知"),
            obj.get("lesson", "先确认关键信息被检索到，再组织回答"),
        )
    except Exception:
        return ("wrong_answer",
                "回答与参考要点不符，可能检索未命中或抽取错误",
                "换更具体的实体/数字词检索，确认命中后再作答")
