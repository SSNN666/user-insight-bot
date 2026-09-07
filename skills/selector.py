"""Skill 选择器 — 场景路由 + 描述检索 + 渐进式披露(Skill engineering 核心)。

两层发现:
1. **确定性场景路由**(调用方决定 group):购物/分析场景的物理隔离是代码决定,
   LLM 不需要做"该进哪个组"的决定 —— 工具绑定名单由注册表元数据推导,
   替代 agent.py 里的硬编码列表(顺带修复 SKILL_ENABLED 开关失效);
2. **描述检索打分**(本模块):按查询关键词对组内 Skill 打分(tags 命中 +
   description 重叠),命中的 Skill 的 SKILL.md 指令才注入上下文 —— 渐进式
   披露,长指令 Skill 不常驻每次对话的 token。

失败兜底:无命中时只披露组内基线 Skill(stats+rules),绑定仍为整组 ——
**选择只影响上下文披露,不影响 LLM 可调用的工具集合**,保证问答质量不回退。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from log.logger import get_logger

logger = get_logger("skill_selector")

# 渐进式披露上限:单条指令截断 + 总块上限(上下文有界)
_MAX_INSTRUCTION_CHARS = 600
_MAX_DISCLOSURE_CHARS = 2400

# 组内基线:无任何命中时仍披露的核心 Skill(保证基本问答能力);
# None = 无命中时披露整个组(小组合用)
_GROUP_FLOOR: dict[str, list[str] | None] = {
    "analysis": ["get_user_segment_stats", "get_segment_rules"],
    "shopping": None,   # 购物组只有 2 个,全组披露
}


@dataclass
class SkillSelection:
    """一次查询的 Skill 选择结果。"""

    group: str                      # shopping | analysis
    bound_names: list[str] = field(default_factory=list)   # 绑定给 LLM 的工具(整组)
    disclosed: list[str] = field(default_factory=list)     # 渐进式披露的 Skill 名
    scores: dict[str, float] = field(default_factory=dict)  # name → 打分
    selected: bool = True           # False=选择器被配置关闭(走旧行为)
    rationale: str = ""             # 一行摘要(日志/Trace)

    def context_header(self) -> str:
        scored = ", ".join(f"{k}={v:g}" for k, v in sorted(self.scores.items()))
        return (f"group={self.group} disclosed={self.disclosed or '-'}"
                + (f" scores={{ {scored} }}" if scored else ""))


# ── 打分(确定性,零 LLM)───────────────────────────────────────


def score_skill(defn, query: str) -> float:
    """对单个 SkillDefinition 打分:tags 命中 1.0/个 + description 子串重叠 0.25/个。"""
    q = query.lower()
    score = 0.0
    for t in defn.tags or []:
        if t and t.lower() in q:
            score += 1.0
    # description 重叠:查询里 ≥2 字的中文片段出现在描述中即计分(粗粒度,确定性)
    desc = (defn.description or "").lower()
    seen: set[str] = set()
    for i in range(len(q) - 1):
        seg = q[i:i + 2]
        if len(seg) == 2 and seg not in seen and seg in desc:
            seen.add(seg)
            score += 0.25
    return round(score, 2)


# ── 选择 ────────────────────────────────────────────────────────


def select_skills(
    group: str,
    query: str,
    *,
    definitions: dict | None = None,
    top_n: int | None = None,
) -> SkillSelection:
    """按场景组 + 查询打分选择 Skill。

    Args:
        group: 确定性路由结果(shopping/analysis),由调用方按 prompt 模板决定;
        query: 原始用户问题;
        definitions: 定义库(默认从 loader 惰性加载,带 mtime 热加载);
        top_n: 披露上限(默认取 settings.SKILL_SELECT_TOP_N)。

    Returns:
        SkillSelection:绑定整组工具 + 披露打分命中的 Skill 指令。
    """
    from config.settings import get_settings
    from skills import SkillRegistry

    settings = get_settings()
    top_n = top_n if top_n is not None else settings.SKILL_SELECT_TOP_N

    bound_names = SkillRegistry.get_enabled_names_by_group(group)
    if not bound_names:
        logger.warning("skill_group_empty", extra={"group": group})

    # ── 描述检索:组内工具 + 组内纯指令 Skill 一起打分 ──
    if definitions is None:
        from skills.loader import load_all_definitions
        definitions = load_all_definitions()
    scores: dict[str, float] = {}
    for name, defn in definitions.items():
        if defn.group != group or not defn.enabled:
            continue
        s = score_skill(defn, query)
        if s > 0:
            scores[name] = s

    # ── 渐进式披露:打分命中 top-N;无命中退到组内基线 ──
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    disclosed = [name for name, _ in ranked[:top_n]]
    if not disclosed:
        floor = _GROUP_FLOOR.get(group)
        if floor is None:
            disclosed = bound_names[:top_n]      # 小组合:全组披露
        else:
            disclosed = [n for n in floor if n in bound_names][:top_n]
        rationale = f"group={group} no_score floor={disclosed}"
    else:
        rationale = (f"group={group} scored={len(scores)} "
                     f"disclosed={disclosed}")

    return SkillSelection(
        group=group,
        bound_names=bound_names,
        disclosed=disclosed,
        scores=scores,
        selected=True,
        rationale=rationale,
    )


def disclosure_block(selection: SkillSelection,
                     definitions: dict | None = None) -> str:
    """把选中的 SKILL.md 指令渲染成 context 块(渐进式披露的产物)。"""
    if not selection.disclosed:
        return ""
    if definitions is None:
        from skills.loader import load_all_definitions
        definitions = load_all_definitions()

    parts = ["[已命中的 Skill 指令(优先遵循,比通用输出要求更具体)]"]
    total = 0
    for name in selection.disclosed:
        defn = definitions.get(name)
        if not defn:
            continue
        body = (defn.instructions or "")[: _MAX_INSTRUCTION_CHARS]
        parts.append(
            f"\n### {name}({defn.group}/{defn.kind})\n{body or defn.description}"
        )
        total += len(body)
        if total > _MAX_DISCLOSURE_CHARS:
            parts.append("\n...(其余 Skill 指令从略)")
            break
    return "\n".join(parts)
