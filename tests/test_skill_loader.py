"""Skill 加载器测试:SKILL.md frontmatter 解析 / 校验 / 注册元数据覆盖 / 热加载。

不依赖真实定义目录(测试用 tmp_path + 显式 defs 注入),注册表用后还原。
"""

import os
import time

import pytest

from skills.loader import (
    SkillDefinitionError, load_all_definitions, parse_skill_file,
    register_from_definitions,
)

VALID_TOOL = """---
name: test_skill
description: 测试 Skill 描述
group: shopping
kind: tool
handler: user_segment.ProductSearchSkill
version: 2.1.0
enabled: true
tags: [搜索, 商品]
---
## 使用说明
这里是指令体。
"""

INSTRUCTION_ONLY = """---
name: report_skill
description: 纯指令 Skill
group: analysis
kind: instruction
tags: [报告]
---
编排 stats/rules 生成报告。
"""


# ── 注册表还原(模块级单例,防止污染其它测试)──────────────────


@pytest.fixture
def _restore_registry():
    from skills import SkillRegistry
    saved = dict(SkillRegistry._skills)
    SkillRegistry.clear()
    yield
    SkillRegistry._skills.clear()
    SkillRegistry._skills.update(saved)


# ── frontmatter 解析 ────────────────────────────────────────────


def test_parse_valid_tool(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text(VALID_TOOL, encoding="utf-8")
    d = parse_skill_file(p)
    assert d.name == "test_skill"
    assert d.group == "shopping"
    assert d.kind == "tool"
    assert d.handler == "user_segment.ProductSearchSkill"
    assert d.version == "2.1.0"
    assert d.tags == ["搜索", "商品"]
    assert "指令体" in d.instructions


def test_parse_instruction_skill_has_no_handler(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text(INSTRUCTION_ONLY, encoding="utf-8")
    d = parse_skill_file(p)
    assert d.kind == "instruction"
    assert d.handler is None          # 纯指令 Skill 不需要 handler


def test_missing_frontmatter_raises(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("没有 frontmatter 的正文", encoding="utf-8")
    with pytest.raises(SkillDefinitionError):
        parse_skill_file(p)


def test_tool_without_handler_raises(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: x\nkind: tool\n---\n", encoding="utf-8")
    with pytest.raises(SkillDefinitionError, match="handler"):
        parse_skill_file(p)


def test_unknown_kind_raises(tmp_path):
    p = tmp_path / "SKILL.md"
    p.write_text("---\nname: x\nkind: magic\n---\n", encoding="utf-8")
    with pytest.raises(SkillDefinitionError, match="kind"):
        parse_skill_file(p)


# ── 目录扫描 + 热加载 ──────────────────────────────────────────


def test_load_all_definitions_and_hot_reload(tmp_path, monkeypatch):
    d1 = tmp_path / "alpha"
    d1.mkdir()
    (d1 / "SKILL.md").write_text(VALID_TOOL, encoding="utf-8")
    monkeypatch.setenv("SKILLS_DEFINITIONS_DIR", str(tmp_path))
    from config.settings import get_settings
    get_settings.cache_clear()

    defs = load_all_definitions(force=True)
    assert "test_skill" in defs

    # 热加载:改动 mtime 后重新加载可见新 Skill(不强制 force)
    d2 = tmp_path / "beta"
    d2.mkdir()
    (d2 / "SKILL.md").write_text(INSTRUCTION_ONLY, encoding="utf-8")
    future = time.time() + 5
    os.utime(d1 / "SKILL.md", (future, future))
    os.utime(d2 / "SKILL.md", (future, future))
    defs2 = load_all_definitions()
    assert "test_skill" in defs2
    assert "report_skill" in defs2
    get_settings.cache_clear()


def test_missing_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLS_DEFINITIONS_DIR", str(tmp_path / "nope"))
    from config.settings import get_settings
    get_settings.cache_clear()
    assert load_all_definitions(force=True) == {}
    get_settings.cache_clear()


# ── 注册:frontmatter 覆盖类属性 ────────────────────────────────


def test_register_applies_metadata(_restore_registry):
    from skills.loader import SkillDefinition
    from skills import SkillRegistry

    defs = {
        "test_skill": SkillDefinition(
            name="test_skill",
            description="定义文件里的描述",
            group="shopping",
            kind="tool",
            handler="user_segment.ProductSearchSkill",
            version="9.9.9",
            tags=["a", "b"],
            instructions="指令",
        ),
    }
    registered = register_from_definitions(defs)
    assert registered == ["test_skill"]

    skill = SkillRegistry.get("test_skill")
    assert skill is not None
    # SKILL.md 是单一事实来源:覆盖了类默认值
    assert skill.group == "shopping"
    assert skill.version == "9.9.9"
    assert skill.tags == ["a", "b"]
    assert skill.description == "定义文件里的描述"
    # 注册表按组可查(物理隔离的数据来源)
    assert "test_skill" in SkillRegistry.get_enabled_names_by_group("shopping")
    assert "test_skill" not in SkillRegistry.get_enabled_names_by_group("analysis")


def test_instruction_skills_not_registered_as_tools(_restore_registry):
    from skills.loader import SkillDefinition

    defs = {
        "report_skill": SkillDefinition(
            name="report_skill", description="纯指令", group="analysis",
            kind="instruction", handler=None, instructions="指令",
        ),
    }
    assert register_from_definitions(defs) == []
    from skills import SkillRegistry
    assert SkillRegistry.get("report_skill") is None   # Skill ≠ Tool 的证据
