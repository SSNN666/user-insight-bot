"""Three-layer rule-based Fact-Check engine.

Zero LLM calls — pure regex + dict lookups, millisecond latency.

Layer 1 — Numerical: Compare claimed numbers vs SkillResult.data.
Layer 2 — Rule: Compare claimed thresholds vs decision tree ground truth.
Layer 3 — Logic: Validate operational conclusions against clustering invariants.

Ground truth source: ``SkillResult.data`` from executed skills.
"""

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from log.logger import get_logger

logger = get_logger(__name__)

# ── Enums ───────────────────────────────────────────────────────


class CheckLayer(str, Enum):
    NUMERICAL = "layer_1"
    RULE = "layer_2"
    LOGIC = "layer_3"


class ViolationSeverity(str, Enum):
    ERROR = "error"      # definite contradiction; blocks in strict mode
    WARNING = "warning"  # possible deviation; annotated in relaxed mode


# ── Data Models ─────────────────────────────────────────────────


class FactViolation(BaseModel):
    layer: CheckLayer
    field: str                           # "用户数", "recency_threshold", etc.
    segment: int | None = None
    claimed_value: Any                   # what the LLM said
    actual_value: Any                    # what SkillResult.data shows
    claimed_text: str = ""               # raw sentence from LLM response
    severity: ViolationSeverity = ViolationSeverity.ERROR
    description: str = ""                # human-readable (Chinese)


class FactCheckResult(BaseModel):
    passed: bool
    violations: list[FactViolation] = Field(default_factory=list)
    check_mode: Literal["relaxed", "strict"] = "relaxed"
    total_checks_run: int = 0
    correction_hints: list[str] = Field(default_factory=list)

    def format_violations_for_user(self) -> str:
        """Relaxed mode: append warnings to the user-visible response."""
        if not self.violations:
            return ""
        lines = ["\n---\n", "## ⚠️ 事实核查警告\n"]
        for i, v in enumerate(self.violations, 1):
            icon = "❌" if v.severity == ViolationSeverity.ERROR else "⚠️"
            lines.append(
                f"{icon} **{v.description}** "
                f"(声称: `{v.claimed_value}`, 实际: `{v.actual_value}`)"
            )
        return "\n".join(lines)

    def format_corrections_for_llm(self) -> str:
        """Strict mode: structured hints for LLM re-generation."""
        if not self.violations:
            return ""
        lines = [
            "[事实核查] 以下内容与数据不符，请重新生成回答并修正：\n",
        ]
        for v in self.violations:
            lines.append(
                f"- {v.field}: 你写的是 `{v.claimed_value}`，"
                f"正确值为 `{v.actual_value}`。{v.description}"
            )
        return "\n".join(lines)


# ── Regex Patterns (compiled once) ──────────────────────────────

RE_SEGMENT_COUNT = re.compile(
    r'分群\s*(\d+)(?:(?!分群\s*\d)(?!\|).)*?(\d+)\s*人'
)
RE_SEGMENT_AVG = re.compile(
    r'分群\s*(\d+)(?:(?!分群\s*\d)(?!\|).)*?平均(近度|频次|消费)'
    r'(?:(?!分群\s*\d)(?!\|).)*?(\d+\.?\d*)'
)
# Markdown table row: | segment | count | avg_recency | avg_freq | avg_monetary |
RE_TABLE_ROW = re.compile(
    r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+\.?\d*)\s*\|\s*(\d+\.?\d*)\s*\|\s*(\d+\.?\d*)\s*\|'
)
RE_THRESHOLD_CLAIM = re.compile(
    r'(recency|近度|频次|frequency|消费|monetary)\s*[<≤>=≥>]+\s*(\d+\.?\d*)'
)
RE_RANGE_CLAIM = re.compile(
    r'(频次|消费|近度|recency|frequency|monetary)\s*在\s*(\d+\.?\d*)\s*[到至和-]\s*(\d+\.?\d*)\s*之间'
)
RE_COMPARATIVE = re.compile(
    r'(高价值|核心|优质).*?(频次|消费|近度).*?(高|多|低|少)'
)
RE_SEGMENT_COMPARE = re.compile(
    r'分群\s*(\d+).*?比.*?分群\s*(\d+).*?(高|低|多|少)'
)


# ── Layer 1: Numerical Check ───────────────────────────────────


def _fact_check_numerical(
    llm_response: str,
    skill_results: list,
) -> list[FactViolation]:
    """Compare claimed numbers against get_user_segment_stats data."""
    violations: list[FactViolation] = []

    # Find stats data in skill_results
    stats_data = None
    for sr in skill_results:
        if hasattr(sr, 'data') and isinstance(sr.data, list) and sr.data:
            first_row = sr.data[0]
            if isinstance(first_row, dict) and 'segment' in first_row and '用户数' in first_row:
                stats_data = sr.data
                break

    if not stats_data:
        logger.debug("no_stats_data_for_numerical_check")
        return violations

    # Build lookup: {segment: {field: value}}
    segment_lookup: dict[int, dict] = {}
    for row in stats_data:
        seg = row.get('segment')
        if seg is not None:
            segment_lookup[int(seg)] = {k: v for k, v in row.items()}

    # Check segment-count claims: "分群X有N人"
    for m in RE_SEGMENT_COUNT.finditer(llm_response):
        seg = int(m.group(1))
        claimed = int(m.group(2))
        actual_row = segment_lookup.get(seg)
        if actual_row:
            actual = actual_row.get('用户数')
            if actual is not None and claimed != actual:
                violations.append(FactViolation(
                    layer=CheckLayer.NUMERICAL,
                    field='用户数',
                    segment=seg,
                    claimed_value=claimed,
                    actual_value=actual,
                    claimed_text=m.group(0),
                    description=f'分群{seg}的用户数为{actual}，而非{claimed}',
                ))

    # Check segment-average claims: "分群X平均频次Y"
    field_map = {'近度': '平均近度', '频次': '平均频次', '消费': '平均消费'}
    for m in RE_SEGMENT_AVG.finditer(llm_response):
        seg = int(m.group(1))
        field_key = field_map.get(m.group(2), m.group(2))
        claimed_str = m.group(3)
        claimed = float(claimed_str)
        actual_row = segment_lookup.get(seg)
        if actual_row:
            actual = actual_row.get(field_key)
            if actual is not None:
                actual_f = float(actual)
                # Smart tolerance: if LLM rounded to integer, check rounding;
                # otherwise use tighter decimal tolerance
                is_match = True
                if '.' not in claimed_str:
                    is_match = round(actual_f) == int(claimed)
                else:
                    is_match = abs(claimed - actual_f) <= 0.01
                if not is_match:
                    violations.append(FactViolation(
                        layer=CheckLayer.NUMERICAL,
                        field=field_key,
                        segment=seg,
                        claimed_value=claimed,
                        actual_value=round(actual_f, 2),
                        claimed_text=m.group(0),
                        description=f'分群{seg}的{field_key}为{actual_f:.2f}，而非{claimed:g}',
                    ))

    # Check markdown table rows
    for m in RE_TABLE_ROW.finditer(llm_response):
        seg = int(m.group(1))
        claimed_count = int(m.group(2))
        actual_row = segment_lookup.get(seg)
        if actual_row:
            actual_count = actual_row.get('用户数')
            if actual_count is not None and claimed_count != actual_count:
                violations.append(FactViolation(
                    layer=CheckLayer.NUMERICAL,
                    field='用户数',
                    segment=seg,
                    claimed_value=claimed_count,
                    actual_value=actual_count,
                    claimed_text=m.group(0),
                    description=f'表格中分群{seg}的用户数为{actual_count}，而非{claimed_count}',
                ))

    return violations


# ── Layer 2: Rule Check ────────────────────────────────────────


def _parse_decision_tree(rules_text: str) -> dict[int, dict[str, tuple[float, float]]]:
    """Parse sklearn export_text output into per-segment threshold bounds.

    Returns: {segment_N: {"recency": (min, max), "frequency": (min, max), "monetary": (min, max)}}
    """
    feature_map = {
        'recency': 'recency', 'frequency': 'frequency', 'monetary': 'monetary',
    }

    bounds: dict[int, dict] = {}
    current_conditions: list[tuple[str, str, float]] = []  # (feature, op, value)

    for line in rules_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Match condition: "|--- feature <= X.XX" (anywhere in line — sklearn export_text uses indentation)
        cond_match = re.search(r'\|-+\s*(\w+)\s*([<>=]+)\s*(\d+\.?\d*)', line)
        if cond_match:
            feature = cond_match.group(1)
            op = cond_match.group(2)
            value = float(cond_match.group(3))
            # sklearn export_text 每行以 '|' 开头,层级 d 的行含 d+1 个 '|'
            # (根 `|---` 计 1 个 → 深度 0)。index('|') 恒为 0,算不出深度,
            # 会误把所有祖先条件都清掉。
            depth = line.count('|') - 1
            # Remove conditions at or beyond this depth
            current_conditions = [c for c in current_conditions if c[3] < depth]
            current_conditions.append((feature, op, value, depth))

        # Match leaf: "|--- class: N" (anywhere in line — sklearn export_text uses indentation)
        leaf_match = re.search(r'\|-*\s*class:\s*(\d+)', line)
        if leaf_match:
            seg = int(leaf_match.group(1))
            seg_bounds: dict[str, tuple[float, float]] = {
                'recency': (0.0, float('inf')),
                'frequency': (0.0, float('inf')),
                'monetary': (0.0, float('inf')),
            }
            for feat, op, val, _d in current_conditions:
                feat = feature_map.get(feat, feat)
                if op == '<=':
                    seg_bounds[feat] = (seg_bounds[feat][0], min(seg_bounds[feat][1], val))
                elif op == '>':
                    seg_bounds[feat] = (max(seg_bounds[feat][0], val), seg_bounds[feat][1])
            bounds[seg] = seg_bounds

    return bounds


def _fact_check_rules(
    llm_response: str,
    skill_results: list,
) -> list[FactViolation]:
    """Compare claimed thresholds against decision tree ground truth."""
    violations: list[FactViolation] = []

    # Find rules data
    rules_text = None
    for sr in skill_results:
        if hasattr(sr, 'data') and isinstance(sr.data, str) and '|---' in sr.data:
            rules_text = sr.data
            break

    if not rules_text:
        logger.debug("no_rules_data_for_rule_check")
        return violations

    try:
        tree_bounds = _parse_decision_tree(rules_text)
    except Exception as e:
        logger.warning("decision_tree_parse_failed", extra={"error": str(e)})
        return violations

    if not tree_bounds:
        return violations

    # Build a set of all thresholds present in the tree
    tree_thresholds: set[float] = set()
    for seg_bounds in tree_bounds.values():
        for fname, (lo, hi) in seg_bounds.items():
            if lo > 0:
                tree_thresholds.add(lo)
            if hi < float('inf'):
                tree_thresholds.add(hi)

    # Check threshold claims
    for m in RE_THRESHOLD_CLAIM.finditer(llm_response):
        claimed = float(m.group(2))
        # Check if this threshold exists in the tree (±1.0 tolerance for colloquial rounding)
        found = any(abs(claimed - t) <= 1.0 for t in tree_thresholds)
        if not found:
            violations.append(FactViolation(
                layer=CheckLayer.RULE,
                field=f"{m.group(1)}_threshold",
                claimed_value=claimed,
                actual_value=f"决策树中无此阈值（现有: {sorted(tree_thresholds)}）",
                claimed_text=m.group(0),
                description=f'阈值 {claimed} 在决策树规则中不存在',
            ))

    return violations


# ── Layer 3: Logic Check ───────────────────────────────────────


def _fact_check_logic(
    llm_response: str,
    skill_results: list,
) -> list[FactViolation]:
    """Validate operational conclusions against clustering invariants."""
    violations: list[FactViolation] = []

    # Find stats data
    stats_data = None
    for sr in skill_results:
        if hasattr(sr, 'data') and isinstance(sr.data, list) and sr.data:
            first_row = sr.data[0]
            if isinstance(first_row, dict) and 'segment' in first_row and '用户数' in first_row:
                stats_data = sr.data
                break

    if not stats_data:
        return violations

    # Derive invariants
    rows_by_seg = sorted(stats_data, key=lambda r: r.get('segment', 0))
    if len(rows_by_seg) < 2:
        return violations

    max_seg = rows_by_seg[-1]
    min_seg = rows_by_seg[0]

    # Invariant: highest segment has highest monetary
    high_monetary_ok = max_seg.get('平均消费', 0) >= min_seg.get('平均消费', 0)

    # Check comparative claims about "高价值用户"
    for m in RE_COMPARATIVE.finditer(llm_response):
        metric_cn = m.group(2)
        direction = m.group(3)
        field_map = {'频次': '平均频次', '消费': '平均消费', '近度': '平均近度'}
        field = field_map.get(metric_cn, metric_cn)

        high_val = max_seg.get(field, 0)
        low_val = min_seg.get(field, 0)

        if direction in ('高', '多') and high_val < low_val:
            violations.append(FactViolation(
                layer=CheckLayer.LOGIC,
                field=field,
                claimed_value=f"高价值用户{metric_cn}{direction}",
                actual_value=f"高价值={high_val}, 低价值={low_val}",
                claimed_text=m.group(0),
                severity=ViolationSeverity.WARNING,
                description=f'高价值用户的{metric_cn}({high_val})实际低于低价值用户({low_val})',
            ))
        elif direction in ('低', '少') and high_val > low_val:
            violations.append(FactViolation(
                layer=CheckLayer.LOGIC,
                field=field,
                claimed_value=f"高价值用户{metric_cn}{direction}",
                actual_value=f"高价值={high_val}, 低价值={low_val}",
                claimed_text=m.group(0),
                severity=ViolationSeverity.WARNING,
                description=f'高价值用户的{metric_cn}({high_val})实际高于低价值用户({low_val})',
            ))

    # Check segment-segment comparisons
    for m in RE_SEGMENT_COMPARE.finditer(llm_response):
        seg_a = int(m.group(1))
        seg_b = int(m.group(2))
        direction = m.group(3)
        # Simple check: do both segments exist?
        seg_a_row = next((r for r in rows_by_seg if r.get('segment') == seg_a), None)
        seg_b_row = next((r for r in rows_by_seg if r.get('segment') == seg_b), None)
        if seg_a_row is None or seg_b_row is None:
            violations.append(FactViolation(
                layer=CheckLayer.LOGIC,
                field='segment_exists',
                claimed_value=f"分群{seg_a} vs 分群{seg_b}",
                actual_value=f"存在分群: {[r['segment'] for r in rows_by_seg]}",
                claimed_text=m.group(0),
                severity=ViolationSeverity.ERROR,
                description=f'提及了不存在的分群',
            ))

    return violations


# ── Orchestrator ────────────────────────────────────────────────


def run_fact_check(
    llm_response: str,
    skill_results: list,
    check_mode: str = "relaxed",
) -> FactCheckResult:
    """Run all three layers of fact-check against SkillResult ground truth.

    Args:
        llm_response: The full text of the LLM's answer.
        skill_results: ``SkillResult`` objects from the current round.
        check_mode: ``"relaxed"`` (annotate) or ``"strict"`` (block + retry).

    Returns:
        ``FactCheckResult`` with violations, correction hints, and pass/fail.
    """
    # Guard: empty reply should not pass silently
    if not llm_response or not llm_response.strip():
        return FactCheckResult(
            passed=False,
            violations=[FactViolation(
                layer=CheckLayer.LOGIC, field="empty_response",
                claimed_value="(empty)", actual_value="non-empty expected",
                severity=ViolationSeverity.ERROR,
                description="LLM 回复为空，无法进行事实核查",
            )],
            check_mode=check_mode,  # type: ignore[arg-type]
            total_checks_run=1,
        )

    all_violations: list[FactViolation] = []

    # Layer 1: Numerical
    all_violations.extend(_fact_check_numerical(llm_response, skill_results))

    # Layer 2: Rule
    all_violations.extend(_fact_check_rules(llm_response, skill_results))

    # Layer 3: Logic
    all_violations.extend(_fact_check_logic(llm_response, skill_results))

    passed = len(all_violations) == 0
    total_checks = len(all_violations)  # each violation = one check that failed

    correction_hints: list[str] = []
    if not passed and check_mode == "strict":
        correction_hints = [
            f"{v.field}: 声称 {v.claimed_value}, 正确值 {v.actual_value}"
            for v in all_violations
        ]

    result = FactCheckResult(
        passed=passed,
        violations=all_violations,
        check_mode=check_mode,  # type: ignore[arg-type]
        total_checks_run=total_checks,
        correction_hints=correction_hints,
    )
    logger.info("fact_check_done", extra={
        "passed": passed,
        "violations": len(all_violations),
        "check_mode": check_mode,
    })
    return result
