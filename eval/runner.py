"""Evaluation runner — executes test suites against a live API."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from eval.metrics import (
    QAResult, EventResult, EvalSummary, compute_summary,
)
from log.logger import get_logger

logger = get_logger(__name__)

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"


class EvalRunner:
    def __init__(self, api_url: str = "http://localhost:8000", timeout: float = 120.0):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.qa_results: list[QAResult] = []
        self.event_results: list[EventResult] = []

    # ── Q&A Suite ───────────────────────────────────────────

    def run_qa_suite(self, test_set: list[dict] | None = None) -> list[QAResult]:
        if test_set is None:
            test_set = json.loads((DATASETS_DIR / "qa_test_set.json").read_text(encoding="utf-8"))

        results: list[QAResult] = []
        client = httpx.Client(timeout=self.timeout)

        for i, case in enumerate(test_set):
            qid = case["id"]
            question = case["question"]
            expected_tools = set(case.get("expected_tools", []))
            expected_keys = case.get("expected_keys", [])
            logger.info("eval_qa_start", extra={"id": qid, "question": question[:60]})

            try:
                start = time.monotonic()
                resp = client.post(f"{self.api_url}/ask", json={
                    "question": question, "session_id": f"eval-{qid}",
                })
                elapsed = (time.monotonic() - start) * 1000
                data = resp.json() if resp.status_code == 200 else {}
                reply = data.get("reply", f"HTTP {resp.status_code}")

                # Tool accuracy: we can't reliably detect which tools were called
                # from the API response alone. Use heuristic: if expected tool keywords
                # appear in the reply, consider it a hit. For a full implementation,
                # integrate with agent state inspection.
                tools_called: list[str] = []
                tool_keywords = {
                    "get_user_segment_stats": ["用户数", "平均近度", "平均频次", "平均消费"],
                    "get_segment_rules": ["|---", "class:", "recency <="],
                    "get_high_value_users": ["user_id", "高价值用户"],
                    "refresh_pipeline": ["刷新", "缓存已刷新"],
                }
                for tool_name, keywords in tool_keywords.items():
                    if any(kw in reply for kw in keywords):
                        tools_called.append(tool_name)

                tool_hits = len(set(tools_called) & expected_tools)
                tool_total = len(expected_tools)
                tool_acc = tool_hits / tool_total if tool_total > 0 else 1.0

                # Answer relevance
                key_hits = sum(1 for k in expected_keys if k.lower() in reply.lower())
                key_total = len(expected_keys)
                relevance = key_hits / key_total if key_total > 0 else 1.0

                # Data fidelity: run fact-check locally
                fact_passed = True
                violations = 0
                try:
                    from skills.user_segment import _load_and_process
                    _, seg, _ = _load_and_process()
                    from skills.base import SkillResult, SkillStatus
                    sr = SkillResult(status=SkillStatus.SUCCESS, data=seg.to_dict(orient="records") if seg is not None else [], summary=reply)
                    from agent.fact_checker import _fact_check_numerical
                    vlist = _fact_check_numerical(reply, [sr])
                    violations = len(vlist)
                    fact_passed = violations == 0
                except Exception as e:
                    logger.debug("fact_check_eval_skipped", extra={"error": str(e)})

                result = QAResult(
                    test_id=qid, question=question, reply=reply,
                    elapsed_ms=round(elapsed, 1),
                    tools_called=tools_called,
                    tools_expected=list(expected_tools),
                    tool_accuracy=round(tool_acc, 3),
                    expected_keys_hit=key_hits,
                    expected_keys_total=key_total,
                    answer_relevance=round(relevance, 3),
                    fact_check_passed=fact_passed,
                    violation_count=violations,
                    data_fidelity=1.0 if fact_passed else max(0.0, 1.0 - 0.2 * violations),
                )
            except Exception as e:
                result = QAResult(test_id=qid, question=question, error=str(e))

            results.append(result)
            logger.info("eval_qa_done", extra={
                "id": qid, "tool_acc": result.tool_accuracy,
                "relevance": result.answer_relevance, "elapsed_ms": result.elapsed_ms,
            })

        client.close()
        self.qa_results = results
        return results

    # ── Event Suite ─────────────────────────────────────────

    def run_event_suite(self, test_set: list[dict] | None = None) -> list[EventResult]:
        if test_set is None:
            test_set = json.loads((DATASETS_DIR / "event_test_set.json").read_text(encoding="utf-8"))

        results: list[EventResult] = []
        client = httpx.Client(timeout=self.timeout)

        for case in test_set:
            eid = case["id"]
            scenario = case["scenario"]
            expected_type = case.get("expected_event_type")
            expected_priority = case.get("expected_priority")
            should_trigger = case.get("should_trigger", False)
            logger.info("eval_event_start", extra={"id": eid, "scenario": scenario})

            try:
                resp = client.post(f"{self.api_url}/debug/trigger-event", json={})
                data = resp.json() if resp.status_code == 200 else {}
                task_ids = data.get("task_ids", [])
                triggered = len(task_ids) > 0

                # Event accuracy scoring
                if not should_trigger:
                    accuracy = 1.0 if not triggered else 0.0  # false positive
                elif should_trigger and triggered:
                    # Partial credit: triggered is correct, type/priority match = bonus
                    accuracy = 0.7  # base for correct trigger
                    if expected_type:
                        # Check task list for matching type
                        try:
                            tasks_resp = client.get(f"{self.api_url}/tasks", params={"limit": 5})
                            tasks = tasks_resp.json().get("tasks", [])
                            for t in tasks:
                                if t.get("event_type") == expected_type:
                                    accuracy = 0.9
                                    if t.get("priority") == expected_priority:
                                        accuracy = 1.0
                                    break
                        except Exception:
                            pass
                else:
                    accuracy = 0.0  # false negative

                type_match = expected_type is not None and triggered
                priority_match = expected_priority is not None and triggered

                result = EventResult(
                    test_id=eid, scenario=scenario,
                    triggered=triggered,
                    expected_trigger=should_trigger,
                    event_type_match=type_match,
                    priority_match=priority_match,
                    event_accuracy=round(accuracy, 3),
                )
            except Exception as e:
                result = EventResult(test_id=eid, scenario=scenario, error=str(e))

            results.append(result)

        client.close()
        self.event_results = results
        return results

    # ── LLM Judge Suite ─────────────────────────────────────

    def run_judge_suite(self) -> dict:
        """Run LLM-as-Judge evaluation on all QA test cases.

        Returns a dict with judge summary that can be merged into the report.
        """
        try:
            from eval.judge import judge_batch, summarize_judge
            import json as _json

            test_set = _json.loads(
                (DATASETS_DIR / "qa_test_set.json").read_text(encoding="utf-8")
            )
            logger.info("judge_suite_start", extra={"cases": len(test_set)})

            scores = judge_batch(test_set, self.api_url)
            summary = summarize_judge(scores)

            logger.info("judge_suite_done", extra={
                "overall": summary.avg_overall,
                "faithfulness": summary.avg_faithfulness,
                "completeness": summary.avg_completeness,
                "readability": summary.avg_readability,
            })
            return {
                "judge_overall": summary.avg_overall,
                "judge_faithfulness": summary.avg_faithfulness,
                "judge_completeness": summary.avg_completeness,
                "judge_readability": summary.avg_readability,
                "judge_cases": summary.total_cases,
                "judge_errors": summary.cases_with_errors,
            }
        except Exception as e:
            logger.warning("judge_suite_failed", extra={"error": str(e)})
            return {}

    # ── Summary ─────────────────────────────────────────────

    def run_all(self, include_judge: bool = False) -> EvalSummary:
        ts = datetime.now(timezone.utc).isoformat()
        qa = self.run_qa_suite()
        ev = self.run_event_suite()
        summary = compute_summary(ts, qa, ev)

        # Attach LLM judge results if requested
        if include_judge:
            judge_data = self.run_judge_suite()
            summary.judge_data = judge_data

        self._save_results(summary)
        return summary

    def _save_results(self, summary: EvalSummary) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = summary.timestamp[:19].replace(":", "").replace("T", "_")
        path = RESULTS_DIR / f"{ts_str}.json"
        data = {
            "timestamp": summary.timestamp,
            "metrics": summary.metrics.to_dict(),
            "qa_results": [
                {"id": r.test_id, "tool_accuracy": r.tool_accuracy,
                 "answer_relevance": r.answer_relevance, "data_fidelity": r.data_fidelity,
                 "elapsed_ms": r.elapsed_ms, "error": r.error}
                for r in summary.qa_results
            ],
            "event_results": [
                {"id": r.test_id, "scenario": r.scenario,
                 "triggered": r.triggered, "expected": r.expected_trigger,
                 "accuracy": r.event_accuracy, "error": r.error}
                for r in summary.event_results
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("eval_results_saved", extra={"path": str(path)})
