"""美食汇记忆：短期 Redis + 长期向量库（Chroma），按用户隔离。

短期（会话）
  - Redis List/Hash：近轮 Q/A、焦点菜
  - 淘汰：条数 LTRIM + Key TTL（默认 24h）

长期（用户）
  - Chroma 向量库：偏好/忌口/失败教训，语义召回
  - 淘汰：每用户上限（默认 200），超出删最旧

上下文组装（ContextAssembler，有预算）
  1) 长期召回（稳定约束，忌口优先）
  2) 会话 working + 近轮对话
  3) 总字符封顶，避免挤占检索证据位
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _safe_user_id(user_id: str) -> str:
    uid = (user_id or "").strip() or "anonymous"
    uid = re.sub(r"[^\w\-]", "_", uid)
    return uid[:64] or "anonymous"


def _store_root() -> Path:
    return Path(__file__).resolve().parents[1] / "memory_store"


# ---------- 短期：Redis ----------

class SessionMemory:
    """同一用户下的会话短期记忆，存 Redis。"""

    def __init__(
        self,
        user_id: str,
        session_id: str | None = None,
        max_turns: int | None = None,
        ttl_sec: int | None = None,
        redis_url: str | None = None,
        persist: bool = True,  # 兼容旧参数；False 时用进程内兜底（评测）
    ):
        self.user_id = _safe_user_id(user_id)
        self.session_id = session_id or str(uuid.uuid4())
        self.max_turns = max_turns or int(os.environ.get("MEISHI_SESSION_MAX_TURNS", "12"))
        self.ttl_sec = ttl_sec if ttl_sec is not None else int(os.environ.get("MEISHI_SESSION_TTL_SEC", "86400"))
        self._mem: dict[str, Any] | None = None  # 进程内兜底
        self._r = None
        if persist:
            self._r = self._connect(redis_url)
        if self._r is None:
            self._mem = {"turns": [], "working": {}}

    def _connect(self, redis_url: str | None):
        host_port = redis_url or os.environ.get("MEISHI_REDIS_MEMORY", "127.0.0.1:6382")
        try:
            import redis
            if host_port.startswith("redis://"):
                return redis.Redis.from_url(host_port, decode_responses=True)
            host, _, port = host_port.partition(":")
            return redis.Redis(host=host or "127.0.0.1", port=int(port or 6382), decode_responses=True)
        except Exception:
            return None

    def _k_turns(self) -> str:
        return f"mem:u:{self.user_id}:s:{self.session_id}:turns"

    def _k_work(self) -> str:
        return f"mem:u:{self.user_id}:s:{self.session_id}:work"

    def _touch_ttl(self) -> None:
        if self._r is None or self.ttl_sec <= 0:
            return
        self._r.expire(self._k_turns(), self.ttl_sec)
        self._r.expire(self._k_work(), self.ttl_sec)

    @property
    def working(self) -> dict[str, str]:
        if self._mem is not None:
            return self._mem["working"]
        assert self._r is not None
        return dict(self._r.hgetall(self._k_work()) or {})

    def add(self, role: str, content: str, **meta) -> None:
        item = json.dumps(
            {"role": role, "content": content, "ts": time.time(), "meta": meta},
            ensure_ascii=False,
        )
        if self._mem is not None:
            self._mem["turns"].append(json.loads(item))
            if len(self._mem["turns"]) > self.max_turns:
                self._mem["turns"] = self._mem["turns"][-self.max_turns :]
            return
        assert self._r is not None
        pipe = self._r.pipeline()
        pipe.rpush(self._k_turns(), item)
        pipe.ltrim(self._k_turns(), -self.max_turns, -1)
        pipe.execute()
        self._touch_ttl()

    def set_focus(self, dish: str | None = None, preference: str | None = None) -> None:
        mapping = {}
        if dish:
            mapping["focus_dish"] = dish
        if preference:
            mapping["preference"] = preference
        if not mapping:
            return
        if self._mem is not None:
            self._mem["working"].update(mapping)
            return
        assert self._r is not None
        self._r.hset(self._k_work(), mapping=mapping)
        self._touch_ttl()

    def recent_turns(self, n: int = 6) -> list[dict]:
        if self._mem is not None:
            return list(self._mem["turns"][-n:])
        assert self._r is not None
        raw = self._r.lrange(self._k_turns(), -n, -1) or []
        out = []
        for x in raw:
            try:
                out.append(json.loads(x))
            except Exception:
                continue
        return out

    def context_text(self, max_chars: int = 1200) -> str:
        parts = [f"用户：{self.user_id}"]
        w = self.working
        if w:
            parts.append("当前会话状态：" + json.dumps(w, ensure_ascii=False))
        for t in self.recent_turns(6):
            parts.append(f"{t.get('role')}: {str(t.get('content', ''))[:400]}")
        return "\n".join(parts)[-max_chars:]

    def rewrite_query(self, query: str) -> str:
        q = query.strip()
        focus = self.working.get("focus_dish")
        if focus and re.search(r"(这道菜|刚才|上面|那个|这个|继续|还要|步骤呢|原料呢)", q):
            if focus not in q:
                return f"{focus} {q}"
        return q


# ---------- 长期：Chroma 向量库 ----------

class LongTermMemory:
    """按用户隔离的长期记忆；Chroma 语义召回 + 每用户条数淘汰。"""

    def __init__(
        self,
        user_id: str,
        persist_dir: Path | None = None,
        max_per_user: int | None = None,
    ):
        self.user_id = _safe_user_id(user_id)
        self.max_per_user = max_per_user or int(os.environ.get("MEISHI_LTM_MAX_PER_USER", "200"))
        root = persist_dir or Path(
            os.environ.get("MEISHI_CHROMA_PATH")
            or str(_store_root() / "chroma_ltm")
        )
        root.mkdir(parents=True, exist_ok=True)
        self._col = None
        self._embedder = None
        self._fallback: list[dict] = []  # Chroma 不可用时退化为内存列表
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(root),
                settings=Settings(anonymized_telemetry=False),
            )
            self._col = client.get_or_create_collection(
                name="meishi_ltm",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self._col = None

    def _get_embedder(self):
        if self._embedder is None:
            from embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._get_embedder().embed(texts)

    def _evict_if_needed(self) -> None:
        """超出每用户上限时删除最旧条目。"""
        if self._col is None:
            if len(self._fallback) > self.max_per_user:
                self._fallback = sorted(self._fallback, key=lambda x: x.get("ts", 0))[-self.max_per_user :]
            return
        try:
            got = self._col.get(where={"user_id": self.user_id}, include=["metadatas"])
            ids = got.get("ids") or []
            metas = got.get("metadatas") or []
            if len(ids) <= self.max_per_user:
                return
            paired = list(zip(ids, metas))
            paired.sort(key=lambda x: float((x[1] or {}).get("ts") or 0))
            drop_n = len(ids) - self.max_per_user
            drop_ids = [i for i, _ in paired[:drop_n]]
            if drop_ids:
                self._col.delete(ids=drop_ids)
        except Exception:
            pass

    def remember_fact(self, text: str, tags: list[str] | None = None) -> None:
        self._add("fact", text=text, tags=tags or [])

    def remember_failure(self, query: str, lesson: str, root_cause: str = "") -> None:
        blob = f"教训：{lesson}。原因：{root_cause}。原问：{query}"
        self._add("failure", text=blob, query=query, lesson=lesson, root_cause=root_cause)

    def _add(self, typ: str, text: str, **extra) -> None:
        mid = f"{self.user_id}:{uuid.uuid4().hex}"
        ts = time.time()
        meta = {
            "user_id": self.user_id,
            "type": typ,
            "ts": ts,
            "tags": ",".join(extra.get("tags") or []) if isinstance(extra.get("tags"), list) else str(extra.get("tags") or ""),
            "query": str(extra.get("query") or "")[:200],
            "lesson": str(extra.get("lesson") or "")[:200],
            "root_cause": str(extra.get("root_cause") or "")[:200],
        }
        if self._col is None:
            self._fallback.append({"id": mid, "text": text, **meta})
            self._evict_if_needed()
            return
        emb = self._embed([text])[0]
        self._col.add(ids=[mid], documents=[text], embeddings=[emb], metadatas=[meta])
        self._evict_if_needed()

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        if self._col is None:
            # 字符重合兜底
            q_chars = set(re.findall(r"[\u4e00-\u9fff]", query))
            scored = []
            for it in self._fallback:
                b_chars = set(re.findall(r"[\u4e00-\u9fff]", it.get("text", "")))
                score = len(q_chars & b_chars)
                if it.get("type") == "failure":
                    score += 0.5
                if score > 0:
                    scored.append((score, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [it for _, it in scored[:top_k]]

        try:
            qv = self._embed([query])[0]
            res = self._col.query(
                query_embeddings=[qv],
                n_results=max(top_k * 3, top_k),
                where={"user_id": self.user_id},
                include=["documents", "metadatas", "distances"],
            )
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            out = []
            for doc, meta, dist in zip(docs, metas, dists):
                out.append({
                    "text": doc,
                    "type": (meta or {}).get("type", "fact"),
                    "ts": (meta or {}).get("ts"),
                    "lesson": (meta or {}).get("lesson", ""),
                    "root_cause": (meta or {}).get("root_cause", ""),
                    "score": 1.0 / (1.0 + float(dist or 0)),
                })
            return out[:top_k]
        except Exception:
            return []

    def context_text(self, query: str, top_k: int = 3, max_chars: int = 800) -> str:
        hits = self.recall(query, top_k=top_k)
        if not hits:
            return ""
        lines = [f"长期记忆（用户 {self.user_id}）："]
        for h in hits:
            if h.get("type") == "failure" or h.get("lesson"):
                lesson = h.get("lesson") or h.get("text", "")
                lines.append(f"- 教训：{lesson}" + (f"（因：{h.get('root_cause','')}）" if h.get("root_cause") else ""))
            else:
                lines.append(f"- 偏好/事实：{h.get('text','')}")
        return "\n".join(lines)[:max_chars]


# ---------- 上下文组装 ----------

@dataclass
class ContextBudget:
    """注入 Agent 的记忆预算（字符近似 token）。"""
    long_term: int = 800
    short_term: int = 1200
    total: int = 2000


class ContextAssembler:
    """
    工业常见组装顺序（记忆侧）：
      [长期约束] → [会话状态/近轮] → 再与系统提示、检索证据、当前问题拼接。
    长期放前面：忌口等硬约束不易被近轮闲聊冲掉；总预算封顶。
    """

    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget(
            long_term=int(os.environ.get("MEISHI_CTX_LTM_CHARS", "800")),
            short_term=int(os.environ.get("MEISHI_CTX_STM_CHARS", "1200")),
            total=int(os.environ.get("MEISHI_CTX_TOTAL_CHARS", "2000")),
        )

    def assemble(self, session: SessionMemory, long_term: LongTermMemory, query: str) -> str:
        ltm = long_term.context_text(query, max_chars=self.budget.long_term)
        stm = session.context_text(max_chars=self.budget.short_term)
        parts = [p for p in (ltm, stm) if p]
        text = "\n\n".join(parts)
        if len(text) > self.budget.total:
            # 超总预算时优先保留长期（截断短期尾部）
            keep_ltm = min(len(ltm), self.budget.long_term)
            rest = self.budget.total - keep_ltm - 2
            text = (ltm[:keep_ltm] + ("\n\n" + stm[: max(0, rest)] if stm and rest > 0 else "")).strip()
        return text


class MemoryHub:
    """用户级记忆入口。"""

    def __init__(
        self,
        user_id: str,
        session_id: str | None = None,
        long_term_path: Path | None = None,  # 兼容旧参数，忽略；改用 Chroma 目录
        persist_session: bool = True,
    ):
        self.user_id = _safe_user_id(user_id)
        self.session = SessionMemory(
            user_id=self.user_id,
            session_id=session_id,
            persist=persist_session,
        )
        self.long_term = LongTermMemory(user_id=self.user_id)
        self.assembler = ContextAssembler()

    def inject_block(self, query: str) -> str:
        return self.assembler.assemble(self.session, self.long_term, query)

    def after_turn(self, query: str, result: dict) -> None:
        self.session.add("user", query)
        self.session.add("assistant", result.get("answer", "")[:800], refused=result.get("refused"))
        hits = result.get("hits") or []
        if not hits:
            for step in result.get("trace") or []:
                r = step.get("result") or {}
                if isinstance(r, dict) and r.get("title") and not r.get("error"):
                    hits = [{"title": r["title"]}]
                    break
                hs = r.get("hits") if isinstance(r, dict) else None
                if hs:
                    hits = hs
                    break
        if hits and not result.get("refused"):
            title = hits[0].get("title") if isinstance(hits[0], dict) else None
            if title:
                self.session.set_focus(dish=title)
        if re.search(r"(不吃|忌口|过敏|不要)", query):
            self.long_term.remember_fact(f"用户偏好：{query}", tags=["preference"])
        if result.get("refused"):
            self.long_term.remember_failure(
                query,
                lesson="域外或证据不足时应拒答，勿编造菜谱",
                root_cause="OOD/低相关检索",
            )
