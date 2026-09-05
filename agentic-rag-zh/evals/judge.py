"""LLM-as-Judge：用与被测不同源的模型做语义判分，补规则评测的假阴性。

规则评测（关键词逐字包含）精度高但召回低——回答正确也可能判错；
Judge 拿【问题/参考答案/候选回答】做语义比对，两者取并集近似真实通过率。
"""
import json
import re

from src import config

JUDGE_SYSTEM = """你是严格的 RAG 问答评测专家。对比【问题】【参考答案】【候选回答】，只输出一个 JSON 对象，不要输出其他内容：
{"semantic_correct": 0或1, "relevancy": 1到5的整数, "reason": "一句话理由"}
判定标准：
- semantic_correct=1：候选回答涵盖了参考答案的关键事实要点（实体/数字/结论），允许措辞不同、表述顺序不同或更详细；
  若关键事实缺失、张冠李戴、数值错误或与参考答案矛盾，为 0。
- relevancy：候选回答与问题的切题程度（5=完全切题，1=答非所问）。"""


def judge_answer(judge_llm, question: str, reference: str, answer: str) -> dict:
    """返回 {semantic_correct, relevancy, reason}；解析失败时带 skipped 标记。"""
    if not answer or answer.startswith("(agent"):
        return {"semantic_correct": 0, "relevancy": 1, "reason": "无有效回答", "skipped": True}
    prompt = f"【问题】{question}\n【参考答案】{reference}\n【候选回答】{answer[:1200]}"
    try:
        resp = judge_llm.chat([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        m = re.search(r"\{.*\}", resp.content or "", re.S)
        obj = json.loads(m.group(0))
        return {
            "semantic_correct": int(bool(obj.get("semantic_correct"))),
            "relevancy": max(1, min(5, int(obj.get("relevancy", 3)))),
            "reason": str(obj.get("reason", ""))[:120],
        }
    except Exception as e:  # judge 挂了不能拖垮整个评测
        return {"semantic_correct": None, "relevancy": None, "reason": f"judge_error: {e}", "skipped": True}


def get_judge_llm():
    """跨源判分：被测用 DeepSeek，judge 默认 GLM。无 key 返回 None（评测退回纯规则）。"""
    if not config.JUDGE_API_KEY:
        return None
    from src.llm.provider import OpenAICompatLLM
    return OpenAICompatLLM(base_url=config.JUDGE_BASE_URL, api_key=config.JUDGE_API_KEY, model=config.JUDGE_MODEL)
