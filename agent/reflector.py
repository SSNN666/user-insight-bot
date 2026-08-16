"""Self-Reflection — validates tool-call results for completeness and consistency.

Uses LLM-as-judge: after tools execute, the reflector inspects each
``SkillResult`` and decides whether the agent has enough information
to answer the original query.
"""

from pydantic import BaseModel, Field

from log.logger import get_logger

logger = get_logger(__name__)


# ── Result types ────────────────────────────────────────────────


class ReflectResult(BaseModel):
    is_complete: bool = True
    has_conflict: bool = False
    missing_info: list[str] = Field(default_factory=list)
    conflict_details: str = ""
    suggestion: str = ""           # 建议：重新调用哪个 Skill
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ── Judge prompt ────────────────────────────────────────────────

REFLECTOR_SYSTEM = """你是一个数据分析质量审核员。检查工具调用结果是否足以回答用户问题，并输出 JSON。

## 判断标准
1. **完整性**: 当前数据能否回答用户问题？缺少什么信息？
2. **一致性**: 多数据源之间是否存在矛盾？
3. **置信度**: 0.0 (完全不可用) ~ 1.0 (完全可信)

## 规则
- 如果缺少关键信息，标记 missing_info 并给出 suggestion（指定需要调用哪个 Skill 或补充什么参数）。
- 如果发现数据冲突（如两个工具返回的同一指标数值不同），标记 has_conflict=true 并在 conflict_details 中说明。
- 如果信息充足且一致，is_complete=true, confidence ≥ 0.8。

## 输出格式（严格 JSON）
{
  "is_complete": true/false,
  "has_conflict": true/false,
  "missing_info": ["缺失描述"],
  "conflict_details": "冲突描述或空字符串",
  "suggestion": "建议或空字符串",
  "confidence": 0.0-1.0
}
"""


# ── Public API ──────────────────────────────────────────────────


def reflect(
    tool_results: list,    # list[SkillResult]
    original_query: str,
    llm,
) -> ReflectResult:
    """Validate the completeness and consistency of tool call results.

    Args:
        tool_results: ``SkillResult`` objects from the current round.
        original_query: The user's original question.
        llm: Chat model for the judgement call.

    Returns:
        ``ReflectResult`` with completeness / conflict / confidence assessment.
    """
    # Build context for the judge
    results_text = "\n---\n".join([
        f"Tool {i+1}: {r.to_context_string()}"
        for i, r in enumerate(tool_results)
    ]) if tool_results else "(no tool results)"

    prompt = f"""原始用户问题: {original_query}

工具调用结果:
{results_text}

请评估上述结果是否足以回答用户问题。"""

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    # Conservative fallback: assume complete to avoid infinite loops
    _DEFAULT = {
        "is_complete": True,
        "has_conflict": False,
        "missing_info": [],
        "conflict_details": "",
        "suggestion": "",
        "confidence": 0.5,
    }

    try:
        response = llm.invoke(messages)
        from common.json_repair import load_json_or_default
        data, repair = load_json_or_default(
            response.content,
            default=_DEFAULT,
            llm=llm, retry_messages=messages,
            hint='{"is_complete":true/false,"has_conflict":true/false,"missing_info":["..."],"conflict_details":"...","suggestion":"...","confidence":0.0-1.0}',
            logger=logger,
        )
        try:
            result = ReflectResult(
                is_complete=data.get("is_complete", True),
                has_conflict=data.get("has_conflict", False),
                missing_info=data.get("missing_info", []),
                conflict_details=data.get("conflict_details", ""),
                suggestion=data.get("suggestion", ""),
                confidence=data.get("confidence", 1.0),
            )
        except Exception as e:
            # confidence 越界等校验失败 → 保守兜底
            logger.warning("reflect_validate_failed", extra={"error": str(e)})
            result = ReflectResult(**_DEFAULT)
        logger.info("reflect_done", extra={
            "is_complete": result.is_complete,
            "has_conflict": result.has_conflict,
            "confidence": result.confidence,
            "repaired": repair.repaired,
            "retries": repair.retries,
        })
        return result

    except Exception as e:
        logger.error("reflect_error", extra={"error": str(e)})
        return ReflectResult(**_DEFAULT)
