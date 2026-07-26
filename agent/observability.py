"""Lightweight LLM observability — tracks usage stats for the Gradio panel.

Collects per-request metrics (tokens, latency, tool calls) in a ring buffer.
Exposes summary stats via API and a Gradio panel.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RequestStats:
    session_id: str
    question: str
    reply_len: int
    duration_ms: float
    tokens: int
    tools_called: list[str]
    fact_check_passed: bool | None
    timestamp: str
    error: str = ""


MAX_RECORDS = 200  # ring buffer cap

_records: deque[RequestStats] = deque(maxlen=MAX_RECORDS)
_lock = threading.Lock()

# Running counters
_total_requests: int = 0
_total_tokens: int = 0
_total_tool_calls: int = 0
_total_errors: int = 0
_total_duration_ms: float = 0.0
_start_time: float = time.time()


def record_request(
    session_id: str,
    question: str,
    reply_len: int,
    duration_ms: float,
    tokens: int,
    tools_called: list[str],
    fact_check_passed: bool | None,
    error: str = "",
) -> None:
    global _total_requests, _total_tokens, _total_tool_calls
    global _total_errors, _total_duration_ms

    stat = RequestStats(
        session_id=session_id,
        question=question[:100],
        reply_len=reply_len,
        duration_ms=round(duration_ms, 1),
        tokens=tokens,
        tools_called=tools_called,
        fact_check_passed=fact_check_passed,
        timestamp=datetime.now(timezone.utc).isoformat(),
        error=error,
    )
    with _lock:
        _records.append(stat)
        _total_requests += 1
        _total_tokens += tokens
        _total_tool_calls += len(tools_called)
        _total_duration_ms += duration_ms
        if error:
            _total_errors += 1


def get_stats() -> dict:
    """Return aggregate observability stats."""
    with _lock:
        records = list(_records)
        uptime = time.time() - _start_time

        if not records:
            return {"uptime_seconds": round(uptime, 0), "total_requests": 0}

        recent = [r for r in records if r.duration_ms > 0]
        if not recent:
            return {"uptime_seconds": round(uptime, 0), "total_requests": 0}

        latencies = [r.duration_ms for r in recent]
        latencies.sort()
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)

        tool_call_rate = (
            sum(1 for r in recent if r.tools_called) / len(recent)
            if recent else 0
        )
        fact_check_pass_rate = (
            sum(1 for r in recent if r.fact_check_passed) / len(recent)
            if recent else 0
        )

        return {
            "uptime_seconds": round(uptime, 0),
            "total_requests": _total_requests,
            "total_tokens": _total_tokens,
            "total_tool_calls": _total_tool_calls,
            "error_count": _total_errors,
            "avg_latency_ms": round(_total_duration_ms / max(_total_requests, 1), 0),
            "p50_latency_ms": latencies[p50_idx],
            "p95_latency_ms": latencies[p95_idx],
            "p99_latency_ms": latencies[p99_idx],
            "avg_tokens_per_request": round(_total_tokens / max(_total_requests, 1), 0),
            "tool_call_rate": round(tool_call_rate, 2),
            "fact_check_pass_rate": round(fact_check_pass_rate, 2),
            "recent_requests": [
                {
                    "session": r.session_id,
                    "question": r.question[:60],
                    "duration_ms": r.duration_ms,
                    "tokens": r.tokens,
                    "tools": r.tools_called,
                    "fact_ok": r.fact_check_passed,
                    "error": r.error[:50] if r.error else "",
                }
                for r in records[-20:]  # last 20
            ],
        }
