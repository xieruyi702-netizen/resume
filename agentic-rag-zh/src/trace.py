"""一次 agent 运行的全程记录：答案、工具调用、检索命中、token 估算。"""
from dataclasses import dataclass, field


@dataclass
class AgentTrace:
    answer: str = ""
    steps: int = 0
    tool_calls: list = field(default_factory=list)   # [{"query","k","n_results"}]
    retrieved: list = field(default_factory=list)    # 各步命中的 doc_id（可重复）
    prompt_chars: int = 0    # token 预算观测：中文字符≈token 量级估算
    completion_chars: int = 0
    error: str | None = None
    plan: dict | None = None  # PAE 模式：规划摘要 {understanding, n_steps, queries, fallback}
