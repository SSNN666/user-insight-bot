"""Dual-layer memory architecture for the agent.

Layer 1 — Chat Session Memory
    Delegated to LangGraph's ``InMemorySaver`` (checkpoint persistence).
    Each ``thread_id`` maps to a conversation history automatically.

Layer 2 — Business Task Memory
    Short-term, in-process memory for autonomous analysis context.
    Tracks which Skills were called, what data was retrieved, and
    intermediate findings. Auto-expires after TTL.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


# ── Layer 2: Business Memory ────────────────────────────────────


@dataclass
class BusinessMemory:
    """Per-session short-term business context.

    Persisted in memory only (not checkpointed). Survives for the
    duration of an autonomous analysis task within one session.
    """

    session_id: str
    called_skills: list[str] = field(default_factory=list)
    retrieved_data: dict[str, Any] = field(default_factory=dict)
    analysis_state: str = "idle"
    findings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add_skill_call(self, skill_name: str) -> None:
        if skill_name not in self.called_skills:
            self.called_skills.append(skill_name)

    def add_finding(self, finding: str) -> None:
        self.findings.append(finding)

    def add_data(self, key: str, value: Any) -> None:
        self.retrieved_data[key] = value

    def format_for_context(self) -> str:
        """Compact summary for injection into LLM system prompt."""
        parts = []
        if self.called_skills:
            parts.append(f"已调用工具: {', '.join(self.called_skills)}")
        if self.findings:
            parts.append("已发现: " + "; ".join(self.findings[-5:]))
        if self.analysis_state != "idle":
            parts.append(f"当前阶段: {self.analysis_state}")
        return "\n".join(parts) if parts else ""


class BusinessMemoryStore:
    """In-memory store for BusinessMemory, with TTL-based expiry."""

    def __init__(self, ttl_seconds: int | None = None):
        settings = get_settings()
        self._store: dict[str, BusinessMemory] = {}
        self.ttl = ttl_seconds or settings.BUSINESS_MEMORY_TTL

    def _expire(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [
            sid for sid, mem in self._store.items()
            if now - mem.created_at > self.ttl
        ]
        for sid in expired:
            del self._store[sid]
            logger.debug("business_memory_expired", extra={"session_id": sid})

    def get_or_create(self, session_id: str) -> BusinessMemory:
        self._expire()
        if session_id not in self._store:
            self._store[session_id] = BusinessMemory(session_id=session_id)
            logger.debug("business_memory_created", extra={"session_id": session_id})
        return self._store[session_id]

    def update(self, session_id: str, **kwargs) -> None:
        mem = self.get_or_create(session_id)
        for k, v in kwargs.items():
            if hasattr(mem, k):
                setattr(mem, k, v)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
        logger.debug("business_memory_cleared", extra={"session_id": session_id})


# ── Module-level singleton ──────────────────────────────────────

_business_store = BusinessMemoryStore()


def get_business_memory(session_id: str) -> BusinessMemory:
    """Get or create business memory for a session."""
    return _business_store.get_or_create(session_id)


def clear_business_memory(session_id: str) -> None:
    """Clear business memory for a session."""
    _business_store.clear(session_id)
