"""Auto-scoring engine — rule-based quality assessment for Q&A samples.

Zero LLM calls. Scores range 0-1. Threshold ≥ 0.5 → positive; < 0.5 → negative.
All scoring weights are configurable via ``config.settings``.
"""

from dataclasses import dataclass, field

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SampleQuality:
    score: float = 0.5
    is_positive: bool = True
    tags: list[str] = field(default_factory=list)


def score_sample(
    question: str,
    reply: str,
    source: str,
    feedback_rating: str | None = None,
    fact_check_passed: bool | None = None,
    violation_count: int = 0,
    tool_rounds: int = 0,
    is_tool_error: bool = False,
) -> SampleQuality:
    """Score a Q&A sample using rule-based heuristics.

    Args:
        question: User question or virtual query.
        reply: Agent's response.
        source: "feedback" | "auto_task" | "manual".
        feedback_rating: "up" | "down" | None.
        fact_check_passed: From Phase 2 FactCheckResult.
        violation_count: Number of FactCheck violations.
        tool_rounds: Tool-call iterations (from auto tasks).
        is_tool_error: True if tool execution threw an exception.

    Returns:
        SampleQuality with score, is_positive, and tags.
    """
    settings = get_settings()
    score = 0.5
    tags: list[str] = []

    # ── 0. Content quality heuristics (Phase: quality-aware scoring) ─

    # 0a. Watcher alert detection — "questions" that are really auto_task
    #     alerts (long, contain "检测到"/"监测窗口") are NOT real user queries.
    alert_markers = ["检测到", "监测窗口", "请综合分析", "请分析可能的原因"]
    is_watcher_alert = (
        source == "auto_task"
        and len(question) > 150
        and any(m in question for m in alert_markers)
    )
    if is_watcher_alert:
        score -= 0.40  # aggressive: these are NOT real Q&A, always push below 0.5 baseline
        tags.append("watcher_alert")

    # 0b. Template answer detection — detect auto_task boilerplate answers.
    #     Only flag answers that are PURELY tables/headers with zero analysis text.
    #     Structured answers with narrative + tables (e.g., manual annotations)
    #     are NOT penalised.
    lines = [l.strip() for l in reply.split("\n") if l.strip()]
    if lines:
        # Count lines that are genuinely narrative (contain Chinese text,
        # numbered lists, bold text — not just table pipes and markdown headers)
        is_content_line = lambda l: (
            bool(l) and not l.startswith("|") and not l.startswith("---")
            and l not in ("", "\\")
        )
        content_lines = [l for l in lines if is_content_line(l)]
        table_lines = [l for l in lines if l.startswith("|")]

        # Only penalise if ZERO content lines (pure table dump)
        if not content_lines and table_lines:
            score -= 0.20
            tags.append("no_narrative")
        # Heavy template: ≤ 2 content lines + ≥ 8 table lines
        elif len(content_lines) <= 2 and len(table_lines) >= 8:
            # Check if the content is just header boilerplate
            is_boilerplate = all(
                l.startswith("###") or l.startswith("**") and "分群" in l
                for l in content_lines
            )
            if is_boilerplate:
                score -= 0.15
                tags.append("template_heavy")

    # 0c. Question quality — penalize too-short/meaningless questions
    if len(question.strip()) < 10:
        score -= 0.25
        tags.append("vague_question")

    # ── 1. Feedback signal (strongest) ──
    if feedback_rating == "up":
        score += settings.SCORER_FEEDBACK_UP_WEIGHT
    elif feedback_rating == "down":
        score += settings.SCORER_FEEDBACK_DOWN_WEIGHT
        tags.append("user_disliked")

    # 2. Fact-check signal
    if fact_check_passed is True:
        score += settings.SCORER_FACT_CHECK_PASSED_WEIGHT
    elif fact_check_passed is False:
        penalty = min(settings.SCORER_FACT_VIOLATION_WEIGHT * violation_count, 0.6)
        score -= penalty
        if violation_count > 0:
            tags.append("fact_violations")

    # 3. Tool efficiency (auto tasks only)
    if source == "auto_task":
        if tool_rounds <= 2 and tool_rounds > 0:
            score += settings.SCORER_EFFICIENT_TOOLS_WEIGHT
            tags.append("efficient")
        elif tool_rounds > 5:
            score += settings.SCORER_INEFFICIENT_TOOLS_WEIGHT
            tags.append("inefficient")

    # 4. Tool error → automatic negative
    if is_tool_error:
        score += settings.SCORER_TOOL_ERROR_WEIGHT
        tags.append("tool_error")

    # 5. Length check
    if len(reply.strip()) < 20:
        score += settings.SCORER_INCOMPLETE_WEIGHT
        tags.append("incomplete")

    # 6. Refusal / error keywords
    refusal_kw = ["抱歉", "无法", "错误", "不支持", "无法回答", "暂无数据"]
    if any(kw in reply for kw in refusal_kw):
        score += settings.SCORER_REFUSAL_WEIGHT
        tags.append("refusal_or_error")

    # 7. Source boost: manual annotations from experts
    if source == "manual":
        score += settings.SCORER_EXPERT_ANNOTATED_WEIGHT
        tags.append("expert_annotated")

    # Clamp and classify
    score = max(0.0, min(1.0, score))
    is_positive = score >= 0.5

    if is_positive and score >= 0.8:
        tags.append("high_quality")
    elif not is_positive and score < 0.3:
        tags.append("low_quality")

    quality = SampleQuality(score=round(score, 3), is_positive=is_positive, tags=tags)
    logger.debug("sample_scored", extra={
        "score": quality.score, "positive": quality.is_positive, "tags": quality.tags,
    })
    return quality
