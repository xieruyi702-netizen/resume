"""智谱 OpenAI 兼容 Chat Completions 客户端。"""
from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.parse
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    p = path or Path(__file__).resolve().parents[1] / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()


class ZhipuChat:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 90,
    ):
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY") or os.environ.get("MEISHI_LLM_KEY", "")
        self.base_url = (base_url or os.environ.get("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
        self.model = model or os.environ.get("MEISHI_GEN_MODEL", "glm-4-flash-250414")
        self.timeout = timeout

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_disabled: bool = True,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("缺少 ZHIPU_API_KEY")
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if thinking_disabled:
            payload["thinking"] = {"type": "disabled"}

        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.netloc
        path = (parsed.path.rstrip("/") or "") + "/chat/completions"
        last_err: Exception | None = None

        for attempt in range(8):
            body_bytes = json.dumps(payload).encode("utf-8")
            try:
                conn = http.client.HTTPSConnection(host, timeout=self.timeout, context=ssl.create_default_context())
                try:
                    conn.request(
                        "POST",
                        path,
                        body=body_bytes,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    )
                    resp = conn.getresponse()
                    raw = resp.read().decode("utf-8", errors="ignore")
                finally:
                    conn.close()

                if resp.status >= 400:
                    if resp.status == 400 and "thinking" in raw.lower() and "thinking" in payload:
                        payload.pop("thinking", None)
                        last_err = RuntimeError(f"HTTP {resp.status}: {raw[:300]}")
                        time.sleep(1.2 * (attempt + 1))
                        continue
                    # 限流：拉长退避
                    if resp.status == 429:
                        last_err = RuntimeError(f"HTTP {resp.status}: {raw[:300]}")
                        time.sleep(min(45.0, 4.0 * (2 ** attempt)))
                        continue
                    raise RuntimeError(f"HTTP {resp.status}: {raw[:300]}")

                body = json.loads(raw)
                choices = body.get("choices") or []
                if not choices:
                    raise RuntimeError(f"empty choices: {json.dumps(body, ensure_ascii=False)[:240]}")
                msg = choices[0].get("message") or {}
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if not str(content).strip():
                    raise RuntimeError(f"empty content: {json.dumps(body, ensure_ascii=False)[:240]}")
                return str(content)
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GLM 调用失败: {last_err}")


def gen_client() -> ZhipuChat:
    return ZhipuChat(model=os.environ.get("MEISHI_GEN_MODEL", "glm-4-flash-250414"))


def judge_client() -> ZhipuChat:
    return ZhipuChat(model=os.environ.get("MEISHI_JUDGE_MODEL", "glm-4.7-flash"))
