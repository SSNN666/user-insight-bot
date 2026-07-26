"""LLM-as-Judge evaluation — semantic quality assessment of agent responses.

Evaluates each agent reply on 3 dimensions (1-5 scale):
  - Faithfulness (忠实度): Does the answer match the data?
  - Completeness (完整性): Does it cover all aspects of the question?
  - Readability (可读性): Is the answer clear, well-structured, useful?

Falls back gracefully if the LLM is unavailable.

Interview pitch: "I built an LLM-as-Judge evaluation pipeline that scores
agent responses on faithfulness, completeness, and readability using a
separate LLM call. This is far more reliable than keyword-matching for
open-ended QA evaluation and directly follows the RAGAS methodology."
"""

import json
import re
from dataclasses import dataclass, field

from log.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JudgeScores:
    """LLM-judge evaluation result for one Q&A pair."""
    test_id: str
    faithfulness: float = 0.0   # 1-5 scale, normalized to 0-1
    completeness: float = 0.0
    readability: float = 0.0
    overall: float = 0.0        # average of the three
    judge_raw: str = ""         # raw LLM response for debugging
    error: str = ""


JUDGE_PROMPT = """你是一个严格但公正的评测专家。请对以下AI回答进行评分。

## 评分维度（每个维度1-5分）

1. **忠实度** (Faithfulness): 回答中的数据是否来自工具查询结果？是否有编造或错误？
   - 5分: 所有数据精确匹配，无编造
   - 3分: 大部分正确，有小错误或遗漏
   - 1分: 严重编造或与数据矛盾

2. **完整性** (Completeness): 回答是否完整覆盖了用户问题的所有方面？
   - 5分: 全面覆盖，超出预期
   - 3分: 基本回答，但有缺失
   - 1分: 严重不完整，答非所问

3. **可读性** (Readability): 回答的结构、格式、语言是否清晰易读？
   - 5分: 格式优美，Markdown规范，逻辑清晰
   - 3分: 基本可读，格式一般
   - 1分: 混乱，难以理解

## 用户问题
{question}

## AI回答
{reply}

## 评分格式
请严格按以下JSON格式输出（只输出JSON，不要其他文字）：
{{"faithfulness": 分数, "completeness": 分数, "readability": 分数, "comment": "一句话总结"}}
"""


def judge_single(question: str, reply: str, test_id: str = "") -> JudgeScores:
    """Evaluate one agent response using LLM-as-Judge.

    Returns JudgeScores with 0-1 normalized values.
    """
    if not reply or len(reply) < 10:
        return JudgeScores(
            test_id=test_id,
            faithfulness=0.0,
            completeness=0.0,
            readability=0.0,
            error="Empty or too-short reply",
        )

    prompt = JUDGE_PROMPT.format(question=question[:500], reply=reply[:2000])

    try:
        from llm.client import get_llm_client
        llm = get_llm_client()
        response = llm.invoke(prompt)
        raw = str(response)

        # Parse JSON from response — use balanced-brace matching
        # (handles nested objects like {"comment": "something with punctuation"})
        start = raw.find("{")
        if start >= 0:
            depth = 0
            end = start
            for i, ch in enumerate(raw[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = raw[start:end]
            try:
                data = json.loads(json_str)
                faith = float(data.get("faithfulness", 3)) / 5.0
                comp = float(data.get("completeness", 3)) / 5.0
                read = float(data.get("readability", 3)) / 5.0
                return JudgeScores(
                    test_id=test_id,
                    faithfulness=round(min(max(faith, 0.0), 1.0), 3),
                    completeness=round(min(max(comp, 0.0), 1.0), 3),
                    readability=round(min(max(read, 0.0), 1.0), 3),
                    overall=round((faith + comp + read) / 3, 3),
                    judge_raw=raw[:500],
                )
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # parse failed; fall through to default scores
    except Exception as e:
        logger.warning("judge_llm_error", extra={"error": str(e), "test_id": test_id})

    return JudgeScores(
        test_id=test_id,
        error="LLM judge unavailable — scores not computed",
    )


def judge_batch(test_cases: list[dict], api_url: str = "http://localhost:8000") -> list[JudgeScores]:
    """Evaluate multiple test cases — calls the live API for each question.

    Args:
        test_cases: List of {"id", "question", ...} dicts.
        api_url: Live FastAPI endpoint to query.

    Returns:
        List of JudgeScores, one per test case.
    """
    import httpx

    results: list[JudgeScores] = []
    client = httpx.Client(timeout=120.0)

    for case in test_cases:
        qid = case.get("id", "unknown")
        question = case.get("question", "")
        logger.info("judge_eval_start", extra={"id": qid})

        try:
            resp = client.post(f"{api_url}/ask", json={
                "question": question,
                "session_id": f"judge-{qid}",
            })
            data = resp.json() if resp.status_code == 200 else {}
            reply = data.get("reply", f"HTTP {resp.status_code}")

            scores = judge_single(question, reply, qid)
            results.append(scores)
            logger.info("judge_eval_done", extra={
                "id": qid, "overall": scores.overall,
                "faithfulness": scores.faithfulness,
            })
        except Exception as e:
            results.append(JudgeScores(test_id=qid, error=str(e)))

    client.close()
    return results


@dataclass
class JudgeSummary:
    """Aggregated judge scores across all test cases."""
    avg_faithfulness: float = 0.0
    avg_completeness: float = 0.0
    avg_readability: float = 0.0
    avg_overall: float = 0.0
    total_cases: int = 0
    cases_with_errors: int = 0
    scores: list[JudgeScores] = field(default_factory=list)


def summarize_judge(scores: list[JudgeScores]) -> JudgeSummary:
    """Aggregate judge scores into a summary."""
    valid = [s for s in scores if not s.error]
    total = len(scores)
    errors = sum(1 for s in scores if s.error)

    if not valid:
        return JudgeSummary(total_cases=total, cases_with_errors=errors, scores=scores)

    return JudgeSummary(
        avg_faithfulness=round(sum(s.faithfulness for s in valid) / len(valid), 3),
        avg_completeness=round(sum(s.completeness for s in valid) / len(valid), 3),
        avg_readability=round(sum(s.readability for s in valid) / len(valid), 3),
        avg_overall=round(sum(s.overall for s in valid) / len(valid), 3),
        total_cases=total,
        cases_with_errors=errors,
        scores=scores,
    )
