"""美食汇记忆：短期 Redis + 长期 Memdir 文件笔记，按用户隔离。

短期（会话）
  - Redis List/Hash：近轮 Q/A、焦点菜、会话滚动摘要
  - 淘汰：条数 LTRIM + Key TTL（默认 24h）

长期（用户）
  - Memdir：MEMORY.md 索引 + 主题 Markdown + PROFILE.md（忌口/偏好）
  - 淘汰：索引行数上限；注入时预算截断

上下文组装（ContextAssembler，有预算）
  1) PROFILE / 相关主题（稳定约束，忌口优先）
  2) 会话 working + 近轮对话
  3) 总字符封顶，避免挤占检索证据位
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_user_id(user_id: str) -> str:
    uid = (user_id or "").strip() or "anonymous"
    uid = re.sub(r"[^\w\-]", "_", uid)
    return uid[:64] or "anonymous"


def _store_root() -> Path:
    env = os.environ.get("MEISHI_MEMORY_STORE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "memory_store"


# ---------- Memdir：文件化长期笔记（对标 Claude Code MEMORY.md）----------

class Memdir:
    """每用户 MEMORY.md 索引 + 主题 Markdown；可审计、可截断注入。"""

    INDEX_MAX_LINES = 40

    def __init__(self, user_id: str):
        self.user_id = _safe_user_id(user_id)
        self.root = _store_root() / "memdir" / self.user_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "MEMORY.md"
        if not self.index_path.exists():
            self.index_path.write_text(
                f"# MEMORY · {self.user_id}\n\n> 索引行数上限 {self.INDEX_MAX_LINES}；细则见主题文件。\n\n",
                encoding="utf-8",
            )

    def _topic_path(self, topic: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", topic)[:40] or "notes"
        return self.root / f"{safe}.md"

    def upsert(self, topic: str, line: str) -> None:
        """追加主题条目，并维护索引一行摘要。"""
        line = (line or "").strip()
        if not line:
            return
        path = self._topic_path(topic)
        if not path.exists():
            path.write_text(f"# {topic}\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(f"- {line}\n")
        # 索引：同 topic 只保留最新一行指针
        idx_lines = self.index_path.read_text(encoding="utf-8").splitlines()
        header, body = [], []
        for ln in idx_lines:
            if ln.startswith("- ["):
                body.append(ln)
            else:
                header.append(ln)
        pointer = f"- [{topic}]({path.name}) · {line[:60]}"
        body = [ln for ln in body if f"[{topic}]" not in ln]
        body.append(pointer)
        body = body[-self.INDEX_MAX_LINES :]
        self.index_path.write_text("\n".join(header + body).rstrip() + "\n", encoding="utf-8")
        if topic in ("allergies", "preferences"):
            self.refresh_profile()

    def refresh_profile(self) -> None:
        """汇总忌口/偏好到 PROFILE.md（对标用户级 CLAUDE.local / 档案）。"""
        lines = [f"# PROFILE · {self.user_id}", ""]
        for topic, title in (("allergies", "忌口/过敏"), ("preferences", "偏好")):
            p = self._topic_path(topic)
            if not p.exists():
                continue
            lines.append(f"## {title}")
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.startswith("- "):
                    lines.append(ln)
            lines.append("")
        (self.root / "PROFILE.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    def profile_text(self, max_chars: int = 400) -> str:
        p = self.root / "PROFILE.md"
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")[:max_chars]

    def index_text(self, max_chars: int = 600) -> str:
        text = self.index_path.read_text(encoding="utf-8") if self.index_path.exists() else ""
        return text[:max_chars]

    def relevant_text(self, query: str, max_topics: int = 3, max_chars: int = 500) -> str:
        """按汉字重合粗选主题文件（轻量 Relevant Memories）。"""
        q_chars = set(re.findall(r"[\u4e00-\u9fff]", query))
        scored: list[tuple[int, Path]] = []
        for p in self.root.glob("*.md"):
            if p.name in ("MEMORY.md", "PROFILE.md"):
                continue
            body = p.read_text(encoding="utf-8")
            b_chars = set(re.findall(r"[\u4e00-\u9fff]", body))
            score = len(q_chars & b_chars)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        parts = []
        for _, p in scored[:max_topics]:
            parts.append(p.read_text(encoding="utf-8")[:400])
        return "\n\n".join(parts)[:max_chars]

    def context_text(self, query: str, max_chars: int = 800) -> str:
        """长期注入块：PROFILE → 相关主题 → 索引摘要。"""
        parts = [
            self.profile_text(max_chars=350),
            self.relevant_text(query, max_topics=3, max_chars=350),
            self.index_text(max_chars=200),
        ]
        text = "\n\n".join(p for p in parts if p)
        if not text:
            return ""
        return f"长期记忆（用户 {self.user_id}）：\n{text}"[:max_chars]


def extract_memories(query: str, result: dict) -> list[tuple[str, str]]:
    """回合末规则抽取（对标 Extract Memories；无额外 LLM 调用）。"""
    out: list[tuple[str, str]] = []
    q = (query or "").strip()
    if re.search(r"(不吃|忌口|过敏|不要|别放)", q):
        out.append(("allergies", f"忌口/约束：{q}"))
    if re.search(r"(喜欢|爱吃|偏好|想吃|常做)", q):
        out.append(("preferences", f"偏好：{q}"))
    if result.get("refused"):
        out.append(("refuse_lessons", f"拒答教训：域外或证据不足勿编造。原问：{q[:80]}"))
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
            out.append(("focus", f"近期焦点菜：{title}"))
    return out


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
        self._set_working(mapping)

    def _set_working(self, mapping: dict[str, str]) -> None:
        if not mapping:
            return
        if self._mem is not None:
            self._mem["working"].update(mapping)
            return
        assert self._r is not None
        self._r.hset(self._k_work(), mapping=mapping)
        self._touch_ttl()

    def set_session_summary(self, summary: str) -> None:
        self._set_working({"session_summary": summary[:500]})

    def maybe_roll_summary(self) -> None:
        """轮次够多时滚一份规则摘要，塞进 working，减轻近轮占位（对标 Session Memory）。"""
        every = int(os.environ.get("MEISHI_SUMMARY_EVERY", "6"))
        turns = self.recent_turns(self.max_turns)
        if len(turns) < every:
            return
        focus = self.working.get("focus_dish", "")
        allergies, prefs, asks = [], [], []
        for t in turns:
            c = str(t.get("content") or "")
            if t.get("role") == "user":
                asks.append(c[:40])
                if re.search(r"(不吃|忌口|过敏|不要|别放)", c):
                    allergies.append(c[:50])
                if re.search(r"(喜欢|爱吃|偏好|想吃)", c):
                    prefs.append(c[:50])
        parts = []
        if focus:
            parts.append(f"焦点菜={focus}")
        if allergies:
            parts.append("忌口=" + "；".join(allergies[-2:]))
        if prefs:
            parts.append("偏好=" + "；".join(prefs[-2:]))
        if asks:
            parts.append("近问=" + " | ".join(asks[-3:]))
        if parts:
            self.set_session_summary("；".join(parts))

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
        if w.get("session_summary"):
            parts.append("会话摘要：" + w["session_summary"])
        # 有摘要时少带近轮，优先摘要（压缩）
        turn_n = 3 if w.get("session_summary") else 6
        rest_w = {k: v for k, v in w.items() if k != "session_summary"}
        if rest_w:
            parts.append("当前会话状态：" + json.dumps(rest_w, ensure_ascii=False))
        for t in self.recent_turns(turn_n):
            parts.append(f"{t.get('role')}: {str(t.get('content', ''))[:400]}")
        return "\n".join(parts)[-max_chars:]

    def rewrite_query(self, query: str) -> str:
        q = query.strip()
        focus = self.working.get("focus_dish")
        if focus and re.search(r"(这道菜|刚才|上面|那个|这个|继续|还要|步骤呢|原料呢)", q):
            if focus not in q:
                return f"{focus} {q}"
        return q


# ---------- 上下文组装 ----------

@dataclass
class ContextBudget:
    """注入 Agent 的记忆预算（字符近似 token）。"""
    long_term: int = 800
    short_term: int = 1200
    total: int = 2000


class ContextAssembler:
    """
    组装顺序：
      [长期 Memdir/PROFILE] → [会话状态/近轮] → 再与系统提示、检索证据、当前问题拼接。
    长期放前面：忌口等硬约束不易被近轮闲聊冲掉；总预算封顶。
    """

    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget(
            long_term=int(os.environ.get("MEISHI_CTX_LTM_CHARS", "800")),
            short_term=int(os.environ.get("MEISHI_CTX_STM_CHARS", "1200")),
            total=int(os.environ.get("MEISHI_CTX_TOTAL_CHARS", "2000")),
        )

    def assemble(self, session: SessionMemory, memdir: Memdir, query: str) -> str:
        ltm = memdir.context_text(query, max_chars=self.budget.long_term)
        stm = session.context_text(max_chars=self.budget.short_term)
        parts = [p for p in (ltm, stm) if p]
        text = "\n\n".join(parts)
        if len(text) > self.budget.total:
            keep_ltm = min(len(ltm), self.budget.long_term)
            rest = self.budget.total - keep_ltm - 2
            text = (ltm[:keep_ltm] + ("\n\n" + stm[: max(0, rest)] if stm and rest > 0 else "")).strip()
        return text


class MemoryHub:
    """用户级记忆入口：Session（Redis）+ Memdir（PROFILE / 主题笔记）。"""

    def __init__(
        self,
        user_id: str,
        session_id: str | None = None,
        long_term_path: Path | None = None,  # 兼容旧参数，已忽略
        persist_session: bool = True,
    ):
        self.user_id = _safe_user_id(user_id)
        self.session = SessionMemory(
            user_id=self.user_id,
            session_id=session_id,
            persist=persist_session,
        )
        self.memdir = Memdir(user_id=self.user_id)
        self.assembler = ContextAssembler()

    def inject_block(self, query: str) -> str:
        """注入：PROFILE / Memdir → 短期（含会话摘要）。"""
        return self.assembler.assemble(self.session, self.memdir, query)

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

        for topic, line in extract_memories(query, result):
            self.memdir.upsert(topic, line)
        self.session.maybe_roll_summary()
