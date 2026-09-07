"""Skill 选择器测试:场景路由 + 描述检索打分 + 渐进式披露 + 兜底。

不依赖真实注册表(monkeypatch 组查询),定义库用显式构造,零 LLM 调用。
"""

from skills.loader import SkillDefinition
from skills.selector import disclosure_block, score_skill, select_skills


def _defn(name, group="analysis", tags=None, desc="描述", instructions="指令体"):
    return SkillDefinition(
        name=name, description=desc, group=group, kind="tool",
        handler="user_segment.UserSegmentStatsSkill",
        tags=tags or [], instructions=instructions,
    )


DEFS = {
    "get_user_segment_stats": _defn(
        "get_user_segment_stats", tags=["分群", "统计", "人数"],
        desc="获取分群人数与 RFM 统计"),
    "get_segment_rules": _defn(
        "get_segment_rules", tags=["规则", "决策树"],
        desc="获取决策树规则"),
    "get_segment_growth": _defn(
        "get_segment_growth", tags=["增长", "环比"],
        desc="环比增长指标"),
    "segment_report": SkillDefinition(
        name="segment_report", description="生成分群分析报告",
        group="analysis", kind="instruction", handler=None,
        tags=["报告", "综合分析"], instructions="依次编排 stats/rules/growth 生成报告"),
    "search_products": _defn(
        "search_products", group="shopping", tags=["商品", "搜索"],
        desc="搜索电商商品"),
}


def _mock_bound(monkeypatch, mapping):
    from skills import SkillRegistry
    monkeypatch.setattr(
        SkillRegistry, "get_enabled_names_by_group",
        classmethod(lambda cls, g: list(mapping.get(g, []))),
    )


def test_scoring_tags_and_description_overlap():
    assert score_skill(DEFS["get_user_segment_stats"], "各分群人数是多少") >= 2.0
    # 描述重叠也能计分(如"环比"不出现在 tags 之外的查询)
    assert score_skill(DEFS["segment_report"], "给我一份分析报告") >= 1.0
    assert score_skill(DEFS["get_segment_rules"], "今天天气怎么样") == 0.0


def test_selection_hits_top_n(monkeypatch):
    _mock_bound(monkeypatch, {"analysis": list(DEFS)})
    sel = select_skills(
        "analysis", "各分群人数和统计情况",
        definitions=DEFS, top_n=2,
    )
    assert sel.group == "analysis"
    assert sel.bound_names == list(DEFS)      # 绑定仍是整组(选择不动工具集合)
    assert "get_user_segment_stats" in sel.disclosed
    assert len(sel.disclosed) <= 2
    assert sel.scores["get_user_segment_stats"] > 0


def test_instruction_skill_disclosed_on_match(monkeypatch):
    """纯指令 Skill(无 handler)也能被披露 —— Skill ≠ Tool 的机制证据。"""
    _mock_bound(monkeypatch, {"analysis": list(DEFS)})
    sel = select_skills("analysis", "给我一份分群分析报告",
                        definitions=DEFS, top_n=3)
    assert "segment_report" in sel.disclosed


def test_no_match_falls_back_to_floor(monkeypatch):
    _mock_bound(monkeypatch, {"analysis": list(DEFS)})
    sel = select_skills("analysis", "今天天气怎么样", definitions=DEFS, top_n=3)
    assert sel.scores == {}
    assert sel.disclosed == ["get_user_segment_stats", "get_segment_rules"]  # 基线


def test_shopping_group_isolated(monkeypatch):
    _mock_bound(monkeypatch, {
        "shopping": ["search_products", "get_categories"],
        "analysis": list(DEFS),
    })
    sel = select_skills("shopping", "有手机吗", definitions=DEFS, top_n=2)
    assert sel.group == "shopping"
    assert "get_user_segment_stats" not in sel.disclosed
    assert "search_products" in sel.disclosed


def test_disclosure_block_contains_instructions():
    from skills.selector import SkillSelection
    sel = SkillSelection(group="analysis", bound_names=[], disclosed=["segment_report"],
                         scores={})
    block = disclosure_block(sel, definitions=DEFS)
    assert "segment_report" in block
    assert "依次编排" in block          # 指令体被披露
    assert "[已命中的 Skill 指令" in block
