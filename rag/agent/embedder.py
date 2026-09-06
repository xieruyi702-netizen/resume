"""向量化：优先本地 bge-small-zh-v1.5；可选智谱 embedding-3。"""
from __future__ import annotations

import http.client
import json
import os
import ssl
import time
import urllib.parse
from pathlib import Path

from llm import load_dotenv

load_dotenv()


class LocalBGEEmbedder:
    """BAAI/bge-small-zh-v1.5（sentence-transformers），512 维。"""

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(self, model_name: str | None = None):
        name = model_name or os.environ.get("MEISHI_LOCAL_EMBED_MODEL") or self.DEFAULT_MODEL
        # 防止误把智谱 API 模型名当成 HF repo
        if name.startswith("embedding-") or name.startswith("glm-"):
            name = self.DEFAULT_MODEL
        self.model_name = name
        self.dimensions = 512
        from sentence_transformers import SentenceTransformer
        # 优先离线读本地缓存，避免评测时反复打 huggingface.co
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str] | str) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        cleaned = [t.replace("\n", " ").strip()[:1200] or " " for t in texts]
        # bge 检索查询侧惯例：加短指令可略提效果；文档侧在建库时可不加
        vecs = self._model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_batched(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self.embed(texts[i : i + batch_size]))
        return out


class ZhipuEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
        self.base_url = (base_url or os.environ.get("ZHIPU_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
        self.model = model or "embedding-3"
        self.dimensions = dimensions or int(os.environ.get("MEISHI_EMBED_DIM", "512"))
        self.timeout = timeout

    def embed(self, texts: list[str] | str) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        if not self.api_key:
            raise RuntimeError("缺少 ZHIPU_API_KEY")
        cleaned = [t.replace("\n", " ").strip()[:1200] or " " for t in texts]
        payload: dict = {"model": self.model, "input": cleaned}
        if self.model.startswith("embedding-3"):
            payload["dimensions"] = self.dimensions
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.netloc
        path = (parsed.path.rstrip("/") or "") + "/embeddings"
        body = json.dumps(payload).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                conn = http.client.HTTPSConnection(host, timeout=self.timeout, context=ssl.create_default_context())
                try:
                    conn.request(
                        "POST",
                        path,
                        body=body,
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
                    raise RuntimeError(f"HTTP {resp.status}: {raw[:300]}")
                data = json.loads(raw)
                items = sorted(data.get("data") or [], key=lambda x: x.get("index", 0))
                if len(items) != len(cleaned):
                    raise RuntimeError(f"embedding 条数不匹配: got {len(items)} expect {len(cleaned)}")
                return [it["embedding"] for it in items]
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Embedding 失败: {last_err}")

    def embed_batched(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self.embed(texts[i : i + batch_size]))
            if i + batch_size < len(texts):
                time.sleep(0.15)
        return out


def get_embedder(prefer: str | None = None):
    """prefer: local | zhipu | auto（默认 auto：local 优先）。"""
    prefer = (prefer or os.environ.get("MEISHI_EMBED_BACKEND", "auto")).lower()
    if prefer in ("zhipu", "api"):
        return ZhipuEmbedder()
    if prefer in ("local", "bge"):
        return LocalBGEEmbedder()
    # auto: try local first
    try:
        return LocalBGEEmbedder()
    except Exception:
        return ZhipuEmbedder()
