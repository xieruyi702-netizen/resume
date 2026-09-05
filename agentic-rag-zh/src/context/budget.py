"""Token 预算表：中文字符 ≈ token 量级，用于观测与裁剪。"""
from src import config

# 字符预算（非精确 tokenizer；够做消融对比）；retrieval/memory 可读配置覆盖
BUDGET = {
    "system_prompt": 1200,
    "failure_fewshot": config.MEMORY_BUDGET_CHARS,
    "plan": 400,
    "retrieval": config.RETRIEVAL_BUDGET_CHARS,
    "output_reserve": 1000,
}


def chars(text: str | None) -> int:
    return len(text or "")


def within_budget(text: str, key: str) -> bool:
    return chars(text) <= BUDGET.get(key, 10**9)
