"""Unit tests for agent/fact_checker.py — all three layers, edge cases."""

import pytest
from agent.fact_checker import (
    _fact_check_numerical,
    _fact_check_rules,
    _fact_check_logic,
    _parse_decision_tree,
    run_fact_check,
    FactCheckResult,
    FactViolation,
    CheckLayer,
    ViolationSeverity,
)
from skills.base import SkillResult, SkillStatus


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

def _make_stats_result(rows: list[dict]) -> SkillResult:
    return SkillResult(
        status=SkillStatus.SUCCESS,
        data=rows,
        summary="stats",
        confidence=0.95,
    )


def _make_rules_result(text: str) -> SkillResult:
    return SkillResult(
        status=SkillStatus.SUCCESS,
        data=text,
        summary="rules",
        confidence=0.9,
    )


# Valid stats fixture: 3 segments with known values
STATS = _make_stats_result([
    {"segment": 0, "用户数": 50, "平均近度": 12.5, "平均频次": 2.1, "平均消费": 89.0},
    {"segment": 1, "用户数": 100, "平均近度": 30.0, "平均频次": 5.3, "平均消费": 245.5},
    {"segment": 2, "用户数": 30, "平均近度": 5.0, "平均频次": 8.7, "平均消费": 520.0},
])

# Valid decision-tree rules text
RULES_TEXT = """|--- recency <= 30.50
|   |--- frequency <= 3.00
|   |   |--- class: 0
|   |--- frequency > 3.00
|   |   |--- monetary <= 300.00
|   |   |   |--- class: 1
|   |   |--- monetary > 300.00
|   |   |   |--- class: 2
|--- recency > 30.50
|   |--- class: 1
"""

RULES = _make_rules_result(RULES_TEXT)


# ══════════════════════════════════════════════════════════════════
# Layer 1 — Numerical
# ══════════════════════════════════════════════════════════════════

class TestNumericalCheck:
    """Layer 1: compare claimed numbers against SkillResult.data."""

    def test_correct_counts_no_violations(self):
        reply = "分群 0 有 50 人，分群 1 有 100 人，分群 2 有 30 人。"
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) == 0

    def test_wrong_segment_count_detected(self):
        reply = "分群 0 有 999 人。"
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) == 1
        v = violations[0]
        assert v.layer == CheckLayer.NUMERICAL
        assert v.field == "用户数"
        assert v.segment == 0
        assert v.claimed_value == 999
        assert v.actual_value == 50

    def test_wrong_average_detected(self):
        reply = "分群 1 的平均频次 999 次。"
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) == 1
        assert violations[0].claimed_value == 999
        assert violations[0].actual_value == 5.3

    def test_rounded_integer_accepted(self):
        """LLM rounding 5.3 → 5 should be accepted (smart tolerance)."""
        reply = "分群 1 的平均频次 5 次。"
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) == 0

    def test_decimal_precision_checked(self):
        """LLM claiming 5.4 when actual is 5.3 should be flagged."""
        reply = "分群 1 的平均频次 5.4 次。"
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) == 1

    def test_nonexistent_segment_ignored(self):
        """Segments not in the data are silently skipped (not in pattern)."""
        # The regex for "分群X有N人" would match "分群9有5人" but segment 9
        # isn't in STATS → no violation generated (can't check unknown segments).
        reply = "分群 9 有 5 人。"
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) == 0  # missing from lookup → skipped

    def test_no_skill_results(self):
        violations = _fact_check_numerical("分群 0 有 50 人", [])
        assert len(violations) == 0  # no data to check against

    def test_markdown_table_row_detected(self):
        reply = """
| 分群 | 用户数 | 平均近度 | 平均频次 | 平均消费 |
|------|--------|----------|----------|----------|
| 0 | 999 | 12.5 | 2.1 | 89.0 |
"""
        violations = _fact_check_numerical(reply, [STATS])
        assert len(violations) >= 1
        assert violations[0].claimed_value == 999
        assert violations[0].actual_value == 50


# 真实数据口径的统计(与线上 JData 3 分群一致)
REAL_STATS = _make_stats_result([
    {"segment": 0, "用户数": 1852, "平均近度": 46.9152, "平均频次": 1.08855, "平均消费": 2478.82},
    {"segment": 1, "用户数": 746, "平均近度": 1.29357, "平均频次": 1.06971, "平均消费": 2591.91},
    {"segment": 2, "用户数": 262, "平均近度": 37.1641, "平均频次": 2.78244, "平均消费": 7678.67},
])


class TestCrossBoundaryRegression:
    """回归:多分群长句里,数字必须归属它自己的分群(线上假阳性案例)。"""

    def test_real_case_multi_segment_prose_no_false_positive(self):
        """线上误报原文:分群2 的 7678.67 被错配给分群0 → 修复后零违规。"""
        reply = ("分群0用户基数最大但活跃度较低；分群1近期活跃度高但频次偏低；"
                 "分群2虽人数最少，但平均消费达7678.67元、频次2.78244次，"
                 "为核心高价值用户群体。")
        violations = _fact_check_numerical(reply, [REAL_STATS])
        assert len(violations) == 0

    def test_count_claim_not_cross_segment(self):
        reply = "分群0用户最多，分群2共262人。"
        violations = _fact_check_numerical(reply, [REAL_STATS])
        assert len(violations) == 0          # 262 归属分群2,核查通过

    def test_wrong_avg_in_first_clause_still_detected(self):
        reply = "分群0的平均消费为9999元；分群2的平均消费达7678.67元。"
        violations = _fact_check_numerical(reply, [REAL_STATS])
        assert len(violations) == 1          # 只罚分群0,不误伤分群2
        v = violations[0]
        assert v.segment == 0
        assert v.claimed_value == 9999
        assert v.actual_value == 2478.82     # 浮点已四舍五入到 2 位

    def test_avg_claim_cannot_cross_table_cells(self):
        """声称与数字之间夹着表格时,不得跨单元格取值。"""
        reply = ("分群0的平均消费为\n"
                 "| 分群 | 用户数 | 平均消费 |\n"
                 "|------|--------|----------|\n"
                 "| 2 | 262 | 7678.67 |\n")
        violations = _fact_check_numerical(reply, [REAL_STATS])
        assert all(v.segment != 0 for v in violations)  # 分群0 未匹配到 7678.67


# ══════════════════════════════════════════════════════════════════
# Layer 2 — Rule
# ══════════════════════════════════════════════════════════════════

class TestRuleCheck:
    """Layer 2: compare claimed thresholds against decision tree."""

    def test_decision_tree_parsing(self):
        bounds = _parse_decision_tree(RULES_TEXT)
        assert len(bounds) >= 2
        # Segment 2 should have recency <= 30.5 and monetary > 300.0
        seg2 = bounds.get(2)
        assert seg2 is not None

    def test_valid_threshold_no_violation(self):
        reply = "recency 阈值大约为 31 天。"
        violations = _fact_check_rules(reply, [RULES])
        # 30.5 rounds to ~31 — within ±1.0 tolerance
        assert len(violations) == 0

    def test_fabricated_threshold_detected(self):
        reply = "recency <= 999 天。"  # regex needs an operator like <=
        violations = _fact_check_rules(reply, [RULES])
        assert len(violations) == 1
        assert violations[0].layer == CheckLayer.RULE

    def test_no_rules_data(self):
        violations = _fact_check_rules("recency <= 30", [STATS])  # STATS has no rules
        assert len(violations) == 0

    def test_malformed_rules_text(self):
        result = _make_rules_result("not a valid tree at all\njust some text")
        violations = _fact_check_rules("recency <= 100", [result])
        assert len(violations) == 0  # graceful degradation


# ══════════════════════════════════════════════════════════════════
# Layer 3 — Logic
# ══════════════════════════════════════════════════════════════════

class TestLogicCheck:
    """Layer 3: validate operational conclusions against invariants."""

    def test_valid_comparison_no_violation(self):
        reply = "高价值用户的消费更高。"
        violations = _fact_check_logic(reply, [STATS])
        # Segment 2 (highest) has 520 monetary > Segment 0's 89 → no violation
        assert len(violations) == 0

    def test_nonexistent_segment_detected(self):
        reply = "分群 9 比 分群 0 高。"
        violations = _fact_check_logic(reply, [STATS])
        assert any(v.field == "segment_exists" for v in violations)

    def test_no_stats_data(self):
        violations = _fact_check_logic("high value users spend more", [])
        assert len(violations) == 0

    def test_single_segment_no_crash(self):
        """Only 1 segment → early return, no crash. Segment comparison skipped."""
        single = _make_stats_result([
            {"segment": 0, "用户数": 10, "平均近度": 5.0, "平均频次": 2.0, "平均消费": 100.0},
        ])
        violations = _fact_check_logic("分群 0 比 分群 1 高", [single])
        # len(rows_by_seg) < 2 → returns early, no crash
        assert len(violations) == 0

    def test_nonexistent_segment_with_enough_data(self):
        """2 real segments + reference to nonexistent seg 3 → flagged."""
        data = _make_stats_result([
            {"segment": 0, "用户数": 10, "平均近度": 5.0, "平均频次": 2.0, "平均消费": 100.0},
            {"segment": 1, "用户数": 20, "平均近度": 15.0, "平均频次": 4.0, "平均消费": 200.0},
        ])
        violations = _fact_check_logic("分群 0 比 分群 9 高", [data])
        assert any(v.field == "segment_exists" for v in violations)


# ══════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════

class TestRunFactCheck:
    """Full three-layer orchestration."""

    def test_all_clean(self):
        reply = "分群 0 有 50 人，平均消费 89 元。分群 1 有 100 人。"
        result = run_fact_check(reply, [STATS, RULES], "relaxed")
        assert result.passed
        assert len(result.violations) == 0

    def test_multiple_violations(self):
        reply = "分群 0 有 999 人，分群 1 平均消费 9999 元。"
        result = run_fact_check(reply, [STATS], "relaxed")
        assert not result.passed
        assert len(result.violations) >= 2

    def test_empty_reply_fails(self):
        result = run_fact_check("", [STATS], "relaxed")
        assert not result.passed

    def test_relaxed_format_warnings(self):
        reply = "分群 0 有 999 人。"
        result = run_fact_check(reply, [STATS], "relaxed")
        warning_text = result.format_violations_for_user()
        assert "⚠️" in warning_text or "事实核查警告" in warning_text

    def test_strict_format_corrections(self):
        reply = "分群 0 有 999 人。"
        result = run_fact_check(reply, [STATS], "strict")
        correction = result.format_corrections_for_llm()
        assert "999" in correction or "50" in correction


# ══════════════════════════════════════════════════════════════════
# 换说法对抗样本(Phase: 覆盖边界收敛)
# 旧版只认"分群X有N人/表格行",这组验证换说法漏检收敛:
# 占比/总数(含千分位)/倍数/差额,以及"消费倍数不误报人数"判别。
# ══════════════════════════════════════════════════════════════════

class TestAdversarialPhrasings:
    """同一事实换四种说法都要检出;正确说法零误报。"""

    def test_pct_share_wrong_detected(self):
        v = _fact_check_numerical("分群2 占比 55%", [STATS])
        assert any(x.field == '用户占比' and x.claimed_value == 55.0 for x in v)

    def test_pct_share_correct_passes(self):
        v = _fact_check_numerical("分群0 占比 28%", [STATS])          # 27.8 四舍五入
        v2 = _fact_check_numerical("分群2 占比 16.7%", [STATS])       # 16.67 同精度
        assert not v and not v2

    def test_total_count_wrong_detected(self):
        v = _fact_check_numerical("全平台共 300 名用户", [STATS])
        assert any(x.field == '用户总数' and x.actual_value == 180 for x in v)

    def test_total_count_correct_passes(self):
        assert not _fact_check_numerical("全平台共 180 名用户", [STATS])

    def test_thousands_separator_count(self):
        """千分位数字也要能核(共 1,800 类长文数字)。"""
        v = _fact_check_numerical("全平台共 1,800 名用户", [STATS])
        assert any(x.field == '用户总数' for x in v)

    def test_multiple_wrong_detected(self):
        v = _fact_check_numerical("分群0 是分群2 的 5 倍", [STATS])   # 实际 1.67
        assert any(x.field == '用户数倍数' for x in v)

    def test_multiple_correct_rounded_passes(self):
        # 50/30=1.667 → "1.7 倍" 是四舍五入,不误报
        assert not _fact_check_numerical("分群0 是分群2 的 1.7 倍", [STATS])

    def test_multiple_metric_not_people_no_false_positive(self):
        """"分群0 的消费是分群2 的 2 倍" —— 说的是金额不是人数,跳过。"""
        assert not _fact_check_numerical("分群0 的消费是分群2 的 2 倍左右", [STATS])

    def test_diff_wrong_direction_detected(self):
        v = _fact_check_numerical("分群0 比 分群2 少 20 人", [STATS])  # 实为多 20
        assert any(x.field == '用户数差额' for x in v)

    def test_diff_wrong_magnitude_detected(self):
        v = _fact_check_numerical("分群1 比 分群0 多 10 人", [STATS])  # 实际差 50
        assert any(x.field == '用户数差额' and x.actual_value == 50 for x in v)

    def test_diff_correct_passes_no_double_flag(self):
        """正确差额零误报,且不得与旧"分群Y有N人"句式重复双报。"""
        assert not _fact_check_numerical("分群0 比 分群2 多 20 人", [STATS])

    def test_direct_count_still_works_after_masking(self):
        """掩码逻辑不影响普通直接陈述句。"""
        assert not _fact_check_numerical("分群0 有 50 人", [STATS])
        v = _fact_check_numerical("分群0 有 60 人", [STATS])
        assert any(x.field == '用户数' for x in v)

    def test_mixed_prose_multiple_claims(self):
        """长散文多句式混合:漏一处抓一处,不整体误报。"""
        prose = ("高价值核心用户集中在分群1(占比约 55%),他们贡献了主要消费;"
                 "分群0 流失风险最高,全平台共 180 名用户中他们占 28%。")
        v = _fact_check_numerical(prose, [STATS])  # 分群1 实际占比 55.6%
        assert not v
        prose_bad = ("高价值核心用户集中在分群1,他们占比约 80%,"
                     "分群0 比 分群2 多 20 人,是平台主力。")
        v2 = _fact_check_numerical(prose_bad, [STATS])
        assert any(x.field == '用户占比' for x in v2)
