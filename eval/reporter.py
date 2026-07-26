"""Report generation — comparison tables and markdown output."""

import json
from pathlib import Path

from eval.metrics import EvalSummary, EvalMetrics

RESULTS_DIR = Path(__file__).parent / "results"


def generate_report(
    current: EvalSummary,
    previous: EvalSummary | None = None,
) -> str:
    """Generate a Markdown comparison report."""
    m = current.metrics
    lines = [
        f"## 评测报告 — {current.timestamp[:19]}",
        "",
        "| 指标 | 本次 | 上次 | 变化 |",
        "|------|------|------|------|",
    ]

    prev_m = previous.metrics if previous else None

    def _delta_row(label: str, key: str, fmt: str = ".3f", lower_is_better: bool = False):
        curr_val = getattr(m, key, 0)
        prev_val = getattr(prev_m, key, 0) if prev_m else None
        curr_str = f"{curr_val:{fmt}}"
        if prev_val is not None:
            delta = curr_val - prev_val
            if lower_is_better:
                arrow = "✅" if delta < 0 else "⚠️" if delta > 0 else "➡️"
            else:
                arrow = "✅" if delta > 0 else "⚠️" if delta < 0 else "➡️"
            prev_str = f"{prev_val:{fmt}}"
            delta_str = f"{delta:+{fmt}} {arrow}"
        else:
            prev_str = "—"
            delta_str = "—"
        return f"| {label} | {curr_str} | {prev_str} | {delta_str} |"

    lines.append(_delta_row("工具调用准确率", "tool_accuracy"))
    lines.append(_delta_row("数据忠实度", "data_fidelity"))
    lines.append(_delta_row("回答业务相关性", "answer_relevance"))
    lines.append(_delta_row("事件触发准确率", "event_accuracy"))
    lines.append(_delta_row("平均响应耗时(ms)", "avg_latency_ms", ".0f", lower_is_better=True))
    lines.append(_delta_row("P50响应耗时(ms)", "p50_latency_ms", ".0f", lower_is_better=True))
    lines.append(_delta_row("P95响应耗时(ms)", "p95_latency_ms", ".0f", lower_is_better=True))

    lines.append("")
    lines.append(f"### 测试覆盖")
    lines.append(f"- Q&A 测试: {m.qa_total} 条 (通过 {m.qa_passed})")
    lines.append(f"- 事件测试: {m.event_total} 条 (通过 {m.event_passed})")

    if m.errors:
        lines.append("")
        lines.append("### 错误记录")
        for e in m.errors[:5]:
            lines.append(f"- {e[:200]}")

    # ── LLM Judge section (if available) ──
    judge = getattr(current, "judge_data", None)
    if judge:
        lines.append("")
        lines.append("### 🤖 LLM-Judge 评测 (语义质量)")
        lines.append("")
        lines.append("| 维度 | 分数 (0-1) | 说明 |")
        lines.append("|------|-----------|------|")
        lines.append(f"| 忠实度 | {judge.get('judge_faithfulness', 0):.3f} | 回答与数据的匹配程度 |")
        lines.append(f"| 完整性 | {judge.get('judge_completeness', 0):.3f} | 是否覆盖所有问题要点 |")
        lines.append(f"| 可读性 | {judge.get('judge_readability', 0):.3f} | 结构和语言质量 |")
        lines.append(f"| **综合** | **{judge.get('judge_overall', 0):.3f}** | 三维度平均 |")
        lines.append(f"")
        lines.append(f"_评测 {judge.get('judge_cases', 0)} 条, {judge.get('judge_errors', 0)} 条出错_")

    lines.append("")
    lines.append(f"_生成时间: {current.timestamp[:19]}_")
    return "\n".join(lines)


def save_report(report_text: str, path: str | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()[:19].replace(":", "").replace("T", "_")
        path = RESULTS_DIR / f"report_{ts}.md"
    else:
        path = Path(path)
    path.write_text(report_text, encoding="utf-8")
    return path


def load_previous_summary(results_dir: str | None = None) -> EvalSummary | None:
    """Load the most recent evaluation result for comparison."""
    rdir = Path(results_dir) if results_dir else RESULTS_DIR
    if not rdir.is_dir():
        return None
    files = sorted(rdir.glob("*.json"), reverse=True)
    if not files:
        return None
    # Skip reports, load latest result
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            from eval.metrics import EvalMetrics, QAResult, EventResult
            metrics = EvalMetrics(**data.get("metrics", {}))
            return EvalSummary(
                timestamp=data.get("timestamp", ""),
                metrics=metrics,
            )
        except Exception:
            continue
    return None
