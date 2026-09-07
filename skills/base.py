"""Pluggable Skill framework — base classes and standardized result envelope.

Each Skill wraps a data-analysis capability with parameter validation,
config-based enable/disable, and a uniform ``SkillResult`` return type.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from log.logger import get_logger

logger = get_logger(__name__)


# ── Status enum ──────────────────────────────────────────────────


class SkillStatus(str, Enum):
    SUCCESS = "success"        # 数据完整，调用成功
    PARTIAL = "partial"        # 部分数据可用
    MISSING = "missing"        # 数据缺失
    CONFLICT = "conflict"      # 多数据源冲突
    ERROR = "error"            # 执行异常


# ── Standardized result ──────────────────────────────────────────


class SkillResult(BaseModel):
    """Every Skill execution returns this envelope.

    The reflector node inspects ``status`` / ``missing_fields`` / ``confidence``
    to decide whether to re-call tools or flag conflicts.
    """

    status: SkillStatus = SkillStatus.SUCCESS
    data: Any = None                       # 成功时的载荷（DataFrame / dict / str）
    summary: str = ""                      # 人类可读摘要
    error: str = ""                        # 异常详情（status=ERROR 时填充）
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_context_string(self) -> str:
        """Compact representation for injection into LLM context."""
        parts = [f"[{self.status.value}] confidence={self.confidence:.0%}"]
        if self.summary:
            parts.append(self.summary[:500])
        if self.error:
            parts.append(f"error: {self.error[:200]}")
        if self.missing_fields:
            parts.append(f"missing: {', '.join(self.missing_fields)}")
        return " | ".join(parts)


# ── Base skill ───────────────────────────────────────────────────


class BaseSkill(ABC):
    """Abstract base for a pluggable analysis skill.

    Subclasses must set class-level attributes and implement ``execute()``.

    Usage::

        class MySkill(BaseSkill):
            name = "my_skill"
            description = "Does something useful."
            input_schema = MyParams  # optional Pydantic model

            def execute(self, **kwargs) -> SkillResult:
                ...
    """

    # ── Class-level metadata (override in subclasses) ────────
    name: str = ""
    description: str = ""
    input_schema: type[BaseModel] | None = None

    # ── Skill engineering metadata ───────────────────────────
    # 由 skills/loader.py 从 SKILL.md frontmatter 覆盖(单一事实来源=定义文件);
    # 类属性仅作为未提供定义文件时的兜底默认值。
    group: str = "analysis"        # 场景组:shopping | analysis
    kind: str = "tool"             # tool=绑 Python 实现 | instruction=纯指令
    version: str = "1.0.0"
    tags: list[str] = []           # 发现用标签(selector 关键词匹配)

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._logger = get_logger(f"skill.{self.name}")

    # ── Subclass contract ────────────────────────────────────

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """Execute the skill and return a standardized result."""
        ...

    # ── LangChain integration ─────────────────────────────────

    def _run(self, **kwargs) -> str:
        """Entry point for LangChain StructuredTool.

        Returns a serialised string representation so the LLM can read it.
        """
        try:
            result = self.execute(**kwargs)
            self._logger.info("skill_executed", extra={
                "skill": self.name,
                "status": result.status.value,
                "confidence": result.confidence,
            })
            return result.to_context_string()
        except Exception as exc:
            self._logger.error("skill_error", extra={
                "skill": self.name,
                "error": str(exc),
            })
            return SkillResult(
                status=SkillStatus.ERROR,
                error=str(exc),
                confidence=0.0,
            ).to_context_string()

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

    def to_langchain_tool(self) -> StructuredTool:
        """Wrap this skill as a LangChain ``StructuredTool``.

        The tool is compatible with LangGraph's ``ToolNode`` and
        can be passed directly to ``llm.bind_tools(...)``.
        """
        schema = self.input_schema if self.input_schema else type(
            f"{self.name}_empty", (BaseModel,), {}
        )
        return StructuredTool.from_function(
            func=self._run,
            coroutine=self._arun,
            name=self.name,
            description=self.description,
            args_schema=schema,
        )

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"<Skill {self.name} ({status})>"
