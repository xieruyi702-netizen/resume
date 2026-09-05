"""LLM 接入层：OpenAI 兼容协议，.env 三行切换厂商。

- OpenAICompatLLM: 真实调用（GLM / DeepSeek / Qwen / OpenAI 均兼容）
- MockLLM: 无 key 时验证全链路 plumbing（第一轮发起 retrieve 工具调用、
  第二轮给出带拒答措辞的最终答案，保证工具循环与评测两条路径都被走到）
"""
import json
import time

from openai import OpenAI

from src import config


class LLMResponse:
    def __init__(self, content=None, tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {}


class OpenAICompatLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.model = model or config.LLM_MODEL
        self.client = OpenAI(
            base_url=base_url or config.LLM_BASE_URL,
            api_key=api_key or config.LLM_API_KEY,
        )

    def chat(self, messages: list[dict], tools: list | None = None) -> LLMResponse:
        last_err: Exception | None = None
        for attempt in range(3):  # 429/超时指数退避重试，批量评测必备
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=config.LLM_TEMPERATURE,
                )
                break
            except Exception as e:  # noqa: BLE001  限流/网络抖动都值得重试
                last_err = e
                if attempt == 2:
                    raise
                wait = 5 * (2 ** attempt)
                print(f"[llm] {type(e).__name__}，{wait}s 后重试（{attempt + 1}/2）", flush=True)
                time.sleep(wait)
        msg = resp.choices[0].message
        tool_calls = [
            {"id": t.id, "name": t.function.name, "arguments": t.function.arguments}
            for t in (msg.tool_calls or [])
        ]
        usage = {}
        if resp.usage:
            usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens}
        return LLMResponse(content=msg.content, tool_calls=tool_calls, usage=usage)


class MockLLM:
    def __init__(self):
        self._n = 0

    def chat(self, messages: list[dict], tools: list | None = None) -> LLMResponse:
        self._n += 1
        if self._n == 1:
            query = (messages[-1]["content"] or "")[:40]
            return LLMResponse(tool_calls=[{
                "id": "mock-call-1",
                "name": "retrieve",
                "arguments": json.dumps({"query": query}, ensure_ascii=False),
            }])
        return LLMResponse(content="（mock 模式）知识库中没有找到相关信息，无法回答该问题。")


def get_llm():
    if config.DRY_RUN or not config.LLM_API_KEY:
        return MockLLM()
    return OpenAICompatLLM()
