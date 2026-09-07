"""Global Skill registry — enables, disables, and queries skills."""

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


class SkillRegistry:
    """Central registry for pluggable skills.

    Skills are registered at import time and can be toggled via
    ``SKILL_ENABLED`` in ``config/settings.py``.

    Usage::

        from skills import SkillRegistry
        from skills.user_segment import UserSegmentStatsSkill

        SkillRegistry.register(UserSegmentStatsSkill())
        tools = SkillRegistry.get_langchain_tools()
    """

    _skills: dict[str, "BaseSkill"] = {}  # noqa: F821

    # ── Registration ─────────────────────────────────────────

    @classmethod
    def register(cls, skill: "BaseSkill") -> None:  # noqa: F821
        """Register a skill. Overwrites if name already exists."""
        cls._skills[skill.name] = skill
        logger.info("skill_registered", extra={"skill_name": skill.name})

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._skills.pop(name, None)

    # ── Queries ──────────────────────────────────────────────

    @classmethod
    def get(cls, name: str) -> "BaseSkill | None":  # noqa: F821
        return cls._skills.get(name)

    @classmethod
    def list_all(cls) -> list["BaseSkill"]:  # noqa: F821
        return list(cls._skills.values())

    @classmethod
    def get_enabled(cls) -> list["BaseSkill"]:  # noqa: F821
        """Return skills that are both configured *and* have ``enabled=True``."""
        settings = get_settings()
        enabled_map = settings.SKILL_ENABLED
        return [
            s for s in cls._skills.values()
            if s.enabled and enabled_map.get(s.name, True)
        ]

    @classmethod
    def get_langchain_tools(cls) -> list:
        """Return LangChain tools for all enabled skills."""
        return [s.to_langchain_tool() for s in cls.get_enabled()]

    # ── 场景分组查询(Skill engineering)─────────────────────
    # 工具绑定按 group 过滤:购物场景只拿到 shopping 组,分析场景只拿
    # analysis 组 —— 物理隔离的元数据来源(替代 agent.py 里硬编码名单)。

    @classmethod
    def get_enabled_by_group(cls, group: str) -> list["BaseSkill"]:  # noqa: F821
        """返回某场景组下 enabled 的 Skill(双层开关:实例 + SKILL_ENABLED 配置)。"""
        return [s for s in cls.get_enabled() if getattr(s, "group", "analysis") == group]

    @classmethod
    def get_enabled_names_by_group(cls, group: str) -> list[str]:
        return [s.name for s in cls.get_enabled_by_group(group)]

    # ── Toggle helpers ───────────────────────────────────────

    @classmethod
    def disable(cls, name: str) -> None:
        skill = cls._skills.get(name)
        if skill:
            skill.enabled = False
            logger.info("skill_disabled", extra={"skill_name": name})

    @classmethod
    def enable(cls, name: str) -> None:
        skill = cls._skills.get(name)
        if skill:
            skill.enabled = True
            logger.info("skill_enabled", extra={"skill_name": name})

    @classmethod
    def clear(cls) -> None:
        cls._skills.clear()
