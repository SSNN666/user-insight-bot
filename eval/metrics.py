"""Core evaluation metrics — 6 dimensions for Q&A and event detection quality."""

import statistics
from dataclasses import dataclass, field


@dataclass
class QAResult:
    test_id: str
    question: str
    reply: str = ""
    elapsed_ms: float = 0.0
    tools_called: list[str] = field(default_factory=list)
    tools_expected: list[str] = field(default_factory=list)
    tool_accuracy: float = 0.0
    expected_keys_hit: int = 0
    expected_keys_total: int = 0
    answer_relevance: float = 0.0
    fact_check_passed: bool = True
    violation_count: int = 0
    data_fidelity: float = 1.0
    error: str = ""


@dataclass
class EventResult:
    test_id: str
    scenario: str
    triggered: bool = False
    expected_trigger: bool = False
    event_type_match: bool = False
    priority_match: bool = False
    event_accuracy: float = 0.0
    error: str = ""


@dataclass
class EvalMetrics:
    tool_accuracy: float = 0.0
    data_fidelity: float = 0.0
    answer_relevance: float = 0.0
    event_accuracy: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    qa_total: int = 0
    qa_passed: int = 0
    event_total: int = 0
    event_passed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool_accuracy": round(self.tool_accuracy, 3),
            "data_fidelity": round(self.data_fidelity, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "event_accuracy": round(self.event_accuracy, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p50_latency_ms": round(self.p50_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "qa_total": self.qa_total,
            "qa_passed": self.qa_passed,
            "event_total": self.event_total,
            "event_passed": self.event_passed,
        }


@dataclass
class EvalSummary:
    timestamp: str
    metrics: EvalMetrics
    qa_results: list[QAResult] = field(default_factory=list)
    event_results: list[EventResult] = field(default_factory=list)
    judge_data: dict = field(default_factory=dict)


def compute_qa_metrics(results: list[QAResult]) -> EvalMetrics:
    """Aggregate Q&A results into summary metrics."""
    if not results:
        return EvalMetrics()

    valid = [r for r in results if not r.error]
    latencies = [r.elapsed_ms for r in valid if r.elapsed_ms > 0]

    return EvalMetrics(
        tool_accuracy=statistics.mean(r.tool_accuracy for r in valid) if valid else 0,
        data_fidelity=statistics.mean(r.data_fidelity for r in valid) if valid else 0,
        answer_relevance=statistics.mean(r.answer_relevance for r in valid) if valid else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=_percentile(latencies, 95) if latencies else 0,
        qa_total=len(results),
        qa_passed=sum(1 for r in valid if r.tool_accuracy >= 0.5 and r.data_fidelity >= 0.5),
        errors=[r.error for r in results if r.error],
    )


def compute_event_metrics(results: list[EventResult]) -> EvalMetrics:
    """Aggregate event detection results."""
    if not results:
        return EvalMetrics()

    valid = [r for r in results if not r.error]
    return EvalMetrics(
        event_accuracy=statistics.mean(r.event_accuracy for r in valid) if valid else 0,
        event_total=len(results),
        event_passed=sum(1 for r in valid if r.event_accuracy >= 0.5),
        errors=[r.error for r in results if r.error],
    )


def compute_summary(
    timestamp: str,
    qa_results: list[QAResult],
    event_results: list[EventResult],
) -> EvalSummary:
    """Combine Q&A and event metrics into a full summary."""
    qa_m = compute_qa_metrics(qa_results)
    ev_m = compute_event_metrics(event_results)

    combined = EvalMetrics(
        tool_accuracy=qa_m.tool_accuracy,
        data_fidelity=qa_m.data_fidelity,
        answer_relevance=qa_m.answer_relevance,
        event_accuracy=ev_m.event_accuracy,
        avg_latency_ms=qa_m.avg_latency_ms,
        p50_latency_ms=qa_m.p50_latency_ms,
        p95_latency_ms=qa_m.p95_latency_ms,
        qa_total=qa_m.qa_total,
        qa_passed=qa_m.qa_passed,
        event_total=ev_m.event_total,
        event_passed=ev_m.event_passed,
        errors=qa_m.errors + ev_m.errors,
    )
    return EvalSummary(timestamp=timestamp, metrics=combined, qa_results=qa_results, event_results=event_results)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f) if f < len(s) - 1 else s[-1]
