"""Agent Trace System — lightweight observability for StateGraph execution.

Records per-node timing, token usage, tool calls, and data flow without
external dependencies. Designed to be consumed by the Gradio Trace panel.

Interview pitch: "I built a lightweight agent observability layer that
visualizes the StateGraph execution — every node transition, tool call,
token cost, and decision point is traced and can be inspected in real-time."
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceSpan:
    """A single node execution span within the agent graph."""
    node_name: str
    started_at: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0
    input_summary: str = ""
    output_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    """Full execution trace for one agent invocation."""
    trace_id: str
    session_id: str
    question: str
    started_at: str
    spans: list[TraceSpan] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    reply: str = ""
    fact_check_passed: bool | None = None
    violation_count: int = 0

    def add_span(self, span: TraceSpan) -> None:
        self.spans.append(span)
        self.total_tokens += span.tokens_used

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "question": self.question[:200],
            "started_at": self.started_at,
            "total_duration_ms": self.total_duration_ms,
            "total_tokens": self.total_tokens,
            "reply": self.reply[:500],
            "fact_check_passed": self.fact_check_passed,
            "violation_count": self.violation_count,
            "spans": [
                {
                    "node": s.node_name,
                    "duration_ms": s.duration_ms,
                    "tokens": s.tokens_used,
                    "input": s.input_summary[:120],
                    "output": s.output_summary[:200],
                    "meta": s.metadata,
                }
                for s in self.spans
            ],
        }

    def format_markdown(self) -> str:
        """Render the trace as a Markdown timeline for Gradio display."""
        # Escape pipe chars in output to avoid breaking the markdown table
        def _esc(text: str) -> str:
            return text.replace("|", "\\|").replace("\n", " ")

        lines = [
            f"## 🔍 Agent 执行链路",
            f"",
            f"**Trace ID**: `{self.trace_id}`  |  **Session**: `{self.session_id}`",
            f"**总耗时**: {self.total_duration_ms:.0f}ms  |  **总Token**: {self.total_tokens}",
            f"**问题**: {self.question[:150]}",
            f"",
            f"| 步骤 | 节点 | 耗时 | Token | 详情 |",
            f"|------|------|------|-------|------|",
        ]

        for i, s in enumerate(self.spans, 1):
            emoji = {
                "preprocess": "🔍", "llm_decide": "🤖", "tools": "🔧",
                "reflect": "🪞", "respond": "💬", "fact_check": "✅",
            }.get(s.node_name, "➡️")

            detail = _esc(s.output_summary[:80])
            if s.metadata:
                if "tool_calls" in s.metadata:
                    tools = ", ".join(s.metadata["tool_calls"])
                    detail = f"调用: {_esc(tools)}"
                elif s.metadata.get("intent"):
                    detail = f"意图: {s.metadata['intent']}"
                elif s.metadata.get("violations") is not None:
                    v = s.metadata['violations']
                    detail = f"违规: {v}个" if v else "通过"

            lines.append(
                f"| {emoji} | **{s.node_name}** | {s.duration_ms:.0f}ms | {s.tokens_used} | {detail} |"
            )

        lines.append("")
        if self.fact_check_passed:
            lines.append("✅ **事实核查通过**")
        elif self.fact_check_passed is False:
            lines.append(f"⚠️ **事实核查未通过** — {self.violation_count} 个违规")
        else:
            lines.append("➡️ **未执行事实核查**")

        return "\n".join(lines)


# ── In-memory trace store (thread-safe) ───────────────────────────

_trace_store: dict[str, AgentTrace] = {}  # session_id → latest trace
_trace_history: list[AgentTrace] = []     # all traces (capped)
_trace_lock = threading.RLock()


def _trace_cap() -> int:
    from config.settings import get_settings
    return get_settings().TRACE_HISTORY_CAP


def start_trace(session_id: str, question: str, trace_id: str = "") -> AgentTrace:
    """Begin a new trace for an agent invocation."""
    if not trace_id:
        trace_id = f"trace-{int(time.time() * 1000)}"
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        question=question,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    with _trace_lock:
        _trace_store[session_id] = trace
    return trace


def record_span(
    session_id: str,
    node_name: str,
    duration_ms: float,
    tokens_used: int = 0,
    input_summary: str = "",
    output_summary: str = "",
    metadata: dict | None = None,
) -> None:
    """Record a node execution span to the active trace."""
    with _trace_lock:
        trace = _trace_store.get(session_id)
        if not trace:
            return
        span = TraceSpan(
            node_name=node_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=round(duration_ms, 1),
            tokens_used=tokens_used,
            input_summary=input_summary[:200],
            output_summary=output_summary[:300],
            metadata=metadata or {},
        )
        trace.add_span(span)


def finish_trace(
    session_id: str,
    reply: str,
    fact_check_passed: bool | None = None,
    violation_count: int = 0,
) -> AgentTrace | None:
    """Complete the trace and archive it."""
    with _trace_lock:
        trace = _trace_store.pop(session_id, None)
        if not trace:
            return None
        trace.reply = reply
        trace.fact_check_passed = fact_check_passed
        trace.violation_count = violation_count
        if trace.spans:
            trace.total_duration_ms = sum(s.duration_ms for s in trace.spans)
        # Archive (cap from settings)
        _trace_history.append(trace)
        cap = _trace_cap()
        while len(_trace_history) > cap:
            _trace_history.pop(0)
    return trace


def get_trace(session_id: str) -> AgentTrace | None:
    """Get the latest trace for a session (active or completed)."""
    with _trace_lock:
        active = _trace_store.get(session_id)
        if active is not None:
            return active
        for t in reversed(_trace_history):
            if t.session_id == session_id:
                return t
    return None


def get_recent_traces(limit: int = 20) -> list[AgentTrace]:
    """Get the most recent completed traces for the history panel."""
    with _trace_lock:
        active = list(_trace_store.values())
        all_traces = _trace_history + active
    return sorted(all_traces, key=lambda t: t.started_at, reverse=True)[:limit]


def clear_traces() -> None:
    """Clear all trace history."""
    with _trace_lock:
        _trace_store.clear()
        _trace_history.clear()
