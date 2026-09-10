"""Three-layer rule-based Fact-Check engine.

Zero LLM calls — pure regex + dict lookups, millisecond latency.

Layer 1 — Numerical: Compare claimed numbers vs SkillResult.data.
Layer 2 — Rule: Compare claimed thresholds vs decision tree ground truth.
Layer 3 — Logic: Validate operational conclusions against clustering invariants.

Ground truth source: ``SkillResult.data`` from executed skills.

诚实边界(README/PORTFOLIO 同步):覆盖句式族 = 直接陈述("分群X有N人"
"平均频次Y")、Markdown 表格行、阈值/区间、"约/左右"整句、以及换说法族:
占比 N%、共/总计 N 人(含千分位)、"X 是 Y 的 N 倍"(带人数词判别,
消费/频次类倍数不误报)、"X 比 Y 多/少 N 人"。规则引擎仍有边界——
更隐蔽的改写(如"高价值用户占了压倒性多数")不在覆盖内,由 relaxed 警告
+ 前端数据源标注兜底,不做"全检"声称。
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
    # 违规条数（不是"执行了多少项核查"）。历史字段名 total_checks_run 名实不符，
    # 已更名以免误导（该字段原先也无外部读取方）。
    total_violations: int = 0
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
# ── 换说法句式族(Phase: 覆盖边界收敛——README 曾自承"只认固定句式")──
# 数字：兼容千分位 "1,200" 与裸写 "1200"。
# ⚠️ 不可写成 \d{1,3}(?:,\d{3})* —— 那样 \d{1,3} 最多吃 3 位，
# "1817" 只匹配到 "181"，后续的 "名用户"/"人" 接不上 → 整个正则不匹配，
# 导致 4 位以上的总数/差额声明被静默跳过（实测漏检 1817/1217 两条）。
_NUM_WITH_SEP = r'\d+(?:,\d{3})*'
_RELAX_WORDS = r'(?:约|大约|大概|近|左右)?'
# 占比:"分群2占比 55%" / "分群 2 占了全部用户的 55.6%"(group2=约/左右等容差词)
RE_PCT_SHARE = re.compile(
    r'分群\s*(\d+)(?:(?!分群\s*\d)(?!\|).){0,20}?'
    r'(?:占比|占了|占总|占全部|占整体|占所有)'
    r'\s*((?:约|大约|大概|近|左右))?\s*(' + _NUM_WITH_SEP + r'\.?\d*)\s*%'
)
# 总数:"全平台共 180 名用户" / "全部用户合计 180 人" / "总计 1,800 人"
# 必须带全局锚定词(全平台/全部用户/整体/总计/总共/合计/平台总计…):
# 裸"共 N 名用户"可能是子集("其中共 262 名 VIP"),不做总数核查
RE_TOTAL_COUNT = re.compile(
    r'(?:全平台|全部用户|全体用户|平台总计|整体|总计|总共|合计|全局)'
    r'(?:用户)?\s*(?:共|有|为|人数为|用户为)?\s*'
    r'(' + _NUM_WITH_SEP + r')\s*(?:名|位)?(?:用户|顾客|会员|人)'
)
# 倍数:"分群0 是分群2 的 2.5 倍" — 判定交给代码:两侧区间含人数词→核查,
# 含指标词(消费/频次等)且无人称词→跳过(防 "消费是…的X倍" 误报)
RE_MULTIPLE_OF = re.compile(
    r'分群\s*(\d+)((?:(?!分群\s*\d)(?!\|).){0,24}?)是\s*分群\s*(\d+)'
    r'((?:(?!分群\s*\d)(?!\|).){0,14}?)的?\s*' + _RELAX_WORDS + r'\s*'
    r'(' + _NUM_WITH_SEP + r'\.?\d*)\s*倍'
)
_PEOPLE_KW = re.compile(r'人数|用户数|成员|规模|体量|用户')
_METRIC_KW = re.compile(r'消费|金额|频次|近度|复购|销售额|营收|客单|monetary|frequency|recency')
# 差额:"分群0 比 分群2 多 70 人"
RE_DIFF_COUNT = re.compile(
    r'分群\s*(\d+)\s*比\s*分群\s*(\d+)\s*(多|少)\s*(' + _NUM_WITH_SEP + r')\s*人'
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

    # 掩码:差额句式("分群X 比分群Y 多 N 人")先行命中并挖空——
    # 否则旧句式会把它内部的 "分群Y … N 人" 当独立人数声明误报。
    _diff_spans = [m.span() for m in RE_DIFF_COUNT.finditer(llm_response)]

    def _blanked(text: str) -> str:
        if not _diff_spans:
            return text
        out = list(text)
        for a, b in _diff_spans:
            out[a:b] = " " * (b - a)
        return "".join(out)

    masked = _blanked(llm_response)

    # Check segment-count claims: "分群X有N人"
    for m in RE_SEGMENT_COUNT.finditer(masked):
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
    for m in RE_SEGMENT_AVG.finditer(masked):
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
    for m in RE_TABLE_ROW.finditer(masked):
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

    total_users = sum(
        (row.get('用户数') or 0) for row in stats_data
    ) or 1

    def _claim_num(raw: str) -> float:
        """千分位串 → float("1,234.5" → 1234.5)。"""
        return float(raw.replace(",", ""))

    def _count_violation(seg: int, field: str, claimed: float,
                         actual: float, text: str, desc: str) -> None:
        violations.append(FactViolation(
            layer=CheckLayer.NUMERICAL, field=field, segment=seg,
            claimed_value=claimed, actual_value=actual,
            claimed_text=text, description=desc,
        ))

    def _rounded_match(claimed_str: str, actual: float) -> bool:
        """按声称精度取整比对:整数声称按四舍五入,小数声称按同位数取整。
        (LLM 写 "1.7 倍" 时实际 1.667 属正常四舍五入,不算编造)"""
        if '.' not in claimed_str:
            return round(actual) == int(_claim_num(claimed_str))
        decimals = len(claimed_str.split('.')[1])
        return round(actual, decimals) == round(_claim_num(claimed_str), decimals)

    # 换说法 1:占比句式 "分群2 占比 55%" → 与 用户数/总数 对照
    # 带"约/大概"等容差词 → ±1 个百分点内放行(人类口语正常,55.6% 可说"约55%")
    for m in RE_PCT_SHARE.finditer(llm_response):
        seg = int(m.group(1))
        actual_row = segment_lookup.get(seg)
        if not actual_row or not actual_row.get('用户数'):
            continue
        relax = bool(m.group(2))
        claimed_str = m.group(3)
        claimed = _claim_num(claimed_str)
        actual_pct = actual_row['用户数'] / total_users * 100.0
        ok = abs(claimed - actual_pct) <= 1.0 if relax \
            else _rounded_match(claimed_str, actual_pct)
        if not ok:
            _count_violation(
                seg, '用户占比', claimed, round(actual_pct, 2), m.group(0),
                f'分群{seg}实际占比为 {actual_pct:.1f}%，而非 {claimed:g}%',
            )

    # 换说法 2:总数句式 "共 180 名用户" → 与 Σ用户数 对照
    for m in RE_TOTAL_COUNT.finditer(llm_response):
        claimed = int(_claim_num(m.group(1)))
        if claimed != total_users:
            violations.append(FactViolation(
                layer=CheckLayer.NUMERICAL, field='用户总数',
                segment=None, claimed_value=claimed,
                actual_value=total_users, claimed_text=m.group(0),
                description=f'全平台用户总数为 {total_users}，而非 {claimed}',
            ))

    # 换说法 3:倍数句式 "分群0 是分群2 的 2.5 倍"
    # 裸"X 倍"（两侧既无指标词也无人称词）指向不明：只按用户数比对会把正确的
    # 指标倍数判成违规（实测误报「分群3 是分群0 的 12 倍」= 正确的消费倍数）。
    # 因此先枚举所有合理解释，命中任一即放行，全不命中才报错。
    _RATIO_FIELDS = ('平均消费', '平均频次', '平均近度')
    for m in RE_MULTIPLE_OF.finditer(llm_response):
        seg_a, seg_b = int(m.group(1)), int(m.group(3))
        span = (m.group(2) or "") + (m.group(4) or "")
        a_row, b_row = segment_lookup.get(seg_a), segment_lookup.get(seg_b)
        if not a_row or not b_row:
            continue
        claimed_str = m.group(5)
        has_metric = bool(_METRIC_KW.search(span))
        has_people = bool(_PEOPLE_KW.search(span))

        candidates: dict[str, float] = {}
        if not has_metric or has_people:      # 未排除人数口径
            if a_row.get('用户数') and b_row.get('用户数'):
                candidates['用户数'] = a_row['用户数'] / b_row['用户数']
        if not has_people or has_metric:      # 未排除指标口径
            for f in _RATIO_FIELDS:
                if a_row.get(f) and b_row.get(f):
                    candidates[f] = a_row[f] / b_row[f]
        if not candidates:
            continue

        if any(_rounded_match(claimed_str, v) for v in candidates.values()):
            continue

        # 全部解释都不匹配 → 报错。
        # 人数口径可用时沿用原字段名与文案（保持既有语义，不改变下游消费方）；
        # 否则退回最接近的指标口径，避免"分群X是分群Y的N倍"被硬扣人数而误报。
        if '用户数' in candidates:
            actual_ratio = candidates['用户数']
            _count_violation(
                seg_a, '用户数倍数', _claim_num(claimed_str), round(actual_ratio, 2),
                m.group(0),
                f'分群{seg_a}用户数是分群{seg_b}的 {actual_ratio:.2f} 倍，'
                f'而非 {_claim_num(claimed_str):g} 倍',
            )
            continue

        best_field, best_val = min(
            candidates.items(),
            key=lambda kv: abs(_claim_num(claimed_str) - kv[1]),
        )
        _count_violation(
            seg_a, f'{best_field}倍数', _claim_num(claimed_str), round(best_val, 2),
            m.group(0),
            f'分群{seg_a}的{best_field}约为分群{seg_b}的 {best_val:.2f} 倍，'
            f'而非 {_claim_num(claimed_str):g} 倍',
        )

    # 换说法 4:差额句式 "分群0 比分群2 多 70 人"
    for m in RE_DIFF_COUNT.finditer(llm_response):
        seg_a, seg_b = int(m.group(1)), int(m.group(2))
        sign = 1 if m.group(3) == '多' else -1
        claimed = int(_claim_num(m.group(4)))
        a_row, b_row = segment_lookup.get(seg_a), segment_lookup.get(seg_b)
        if not a_row or not b_row or a_row.get('用户数') is None or b_row.get('用户数') is None:
            continue
        actual_diff = a_row['用户数'] - b_row['用户数']
        if sign * actual_diff <= 0:  # 方向性错误(说多实少)
            _count_violation(
                seg_a, '用户数差额', f"{m.group(3)}{claimed}",
                actual_diff, m.group(0),
                f'分群{seg_a}比{seg_b}{"少" if actual_diff < 0 else "多"}'
                f'{abs(actual_diff)} 人，而非{m.group(3)}{claimed} 人',
            )
        elif abs(actual_diff) != claimed:
            _count_violation(
                seg_a, '用户数差额', f"{m.group(3)}{claimed}",
                actual_diff, m.group(0),
                f'分群{seg_a}与{seg_b}人数差为 {abs(actual_diff)}，而非 {claimed}',
            )

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

    # 注：原有一行 high_monetary_ok = max_seg['平均消费'] >= min_seg['平均消费']
    # 计算后从未被使用（死变量，且易被误读为"已校验该不变量"）。已移除。
    # 下方的方向核查直接取 max_seg/min_seg 的实际值比对，不依赖该不变量成立。

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
            total_violations=1,
        )

    all_violations: list[FactViolation] = []

    # Layer 1: Numerical
    all_violations.extend(_fact_check_numerical(llm_response, skill_results))

    # Layer 2: Rule
    all_violations.extend(_fact_check_rules(llm_response, skill_results))

    # Layer 3: Logic
    all_violations.extend(_fact_check_logic(llm_response, skill_results))

    passed = len(all_violations) == 0

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
        total_violations=len(all_violations),
        correction_hints=correction_hints,
    )
    logger.info("fact_check_done", extra={
        "passed": passed,
        "violations": len(all_violations),
        "check_mode": check_mode,
    })
    return result
