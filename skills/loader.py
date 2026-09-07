"""声明式 Skill 定义加载器 — SKILL.md 文件系统即声明,加载器注册即生效。

格式(每个 Skill 一个目录,``SKILL.md`` 含 YAML frontmatter + Markdown 指令):

```markdown
---
name: get_user_segment_stats
description: 获取各用户分群的人数、平均近度、频次、平均消费金额
group: analysis
kind: tool
handler: user_segment.UserSegmentStatsSkill
version: 1.2.0
tags: [rfm, cluster, stats]
---
## 使用说明
...Markdown 指令体(渐进式披露时注入 LLM 上下文)...
```

设计要点:
- **单一事实来源**:frontmatter 的 description/group/tags 注册时覆盖类属性,
  改定义文件即改行为,不碰 Python 代码;
- **kind=tool**:需 ``handler`` 点路径,加载器实例化后注册进 ``SkillRegistry``;
- **kind=instruction**:纯指令 Skill(无 handler,不注册为工具),留在定义库供
  ``skills/selector.py`` 渐进式披露 —— Skill ≠ Tool 的实物证明;
- **热加载**:``load_all_definitions()`` 按文件 mtime 惰性重扫,定义层改动
  即时生效(工具绑定名单在 import 时快照,新增 Skill 需重启后进入 LLM 工具集)。
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from log.logger import get_logger

logger = get_logger("skill_loader")

# ── 默认定义目录:项目根/skills/definitions ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_definitions_dir() -> Path:
    """SKILL.md 定义目录(可用 SKILLS_DEFINITIONS_DIR 覆盖,用于测试/外部挂载)。"""
    from config.settings import get_settings
    return Path(get_settings().SKILLS_DEFINITIONS_DIR).resolve()


class SkillDefinitionError(Exception):
    """SKILL.md 解析/校验失败。"""


# ── 定义模型 ────────────────────────────────────────────────────


class SkillDefinition(BaseModel):
    """一份 SKILL.md 的解析结果(唯一合法来源)。"""

    name: str
    description: str = ""
    group: str = "analysis"          # shopping | analysis(未来可扩展)
    kind: str = "tool"               # tool | instruction
    handler: str | None = None       # kind=tool 必填:点路径定位 BaseSkill 类
    version: str = "1.0.0"
    enabled: bool = True
    tags: list[str] = Field(default_factory=list)
    instructions: str = ""           # frontmatter 之后的 Markdown 指令体
    source_file: str = ""
    mtime: float = 0.0

    def short(self) -> str:
        return (
            f"{self.name}[{self.group}/{self.kind}] v{self.version} "
            f"tags={self.tags or '-'}"
        )


# ── Frontmatter 解析 ────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)


def parse_skill_file(path: str | Path) -> SkillDefinition:
    """解析单个 SKILL.md → SkillDefinition;任何缺失/非法字段都抛 SkillDefinitionError。"""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise SkillDefinitionError(
            f"{p.name}: 缺少 YAML frontmatter(需以 '---\\n...\\n---' 开头)")
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        raise SkillDefinitionError(f"{p.name}: frontmatter YAML 解析失败: {e}")
    if not isinstance(meta, dict):
        raise SkillDefinitionError(f"{p.name}: frontmatter 必须是键值映射")

    name = str(meta.get("name", "")).strip()
    if not name:
        raise SkillDefinitionError(f"{p.name}: 缺少必填字段 name")

    kind = str(meta.get("kind", "tool")).strip()
    if kind not in ("tool", "instruction"):
        raise SkillDefinitionError(
            f"{p.name}: kind 只能是 tool|instruction,实际 '{kind}'")

    handler = meta.get("handler")
    if kind == "tool" and not handler:
        raise SkillDefinitionError(f"{p.name}: kind=tool 必须提供 handler 点路径")

    group = str(meta.get("group", "analysis")).strip()
    tags = meta.get("tags", [])
    if not isinstance(tags, list):
        raise SkillDefinitionError(f"{p.name}: tags 必须是列表")

    body = text[m.end():].strip()
    try:
        return SkillDefinition(
            name=name,
            description=str(meta.get("description", "")).strip(),
            group=group,
            kind=kind,
            handler=str(handler).strip() if handler else None,
            version=str(meta.get("version", "1.0.0")).strip(),
            enabled=bool(meta.get("enabled", True)),
            tags=[str(t) for t in tags],
            instructions=body,
            source_file=str(p),
            mtime=p.stat().st_mtime,
        )
    except ValidationError as e:
        raise SkillDefinitionError(f"{p.name}: 定义校验失败: {e}")


# ── 定义库(进程级,带 mtime 热加载)──────────────────────────────

_lock = threading.RLock()
_definitions: dict[str, SkillDefinition] | None = None
_scanned_at: float = 0.0


def _scan_dir(skills_dir: Path) -> dict[str, SkillDefinition]:
    """全量扫描目录下所有 SKILL.md(单文件失败不阻断其余,记日志)。"""
    found: dict[str, SkillDefinition] = {}
    if not skills_dir.is_dir():
        logger.warning("skills_dir_missing", extra={"dir": str(skills_dir)})
        return found
    for md in sorted(skills_dir.rglob("SKILL.md")):
        try:
            defn = parse_skill_file(md)
            if defn.name in found:
                logger.warning("skill_duplicate_name", extra={"name": defn.name})
            found[defn.name] = defn
        except SkillDefinitionError as e:
            logger.error("skill_definition_invalid", extra={"file": str(md), "error": str(e)})
    return found


def load_all_definitions(force: bool = False) -> dict[str, SkillDefinition]:
    """返回 name → SkillDefinition(带 mtime 惰性重扫;force=True 强制全量重扫)。"""
    global _definitions, _scanned_at
    with _lock:
        if _definitions is None or force:
            _definitions = _scan_dir(default_definitions_dir())
            _scanned_at = 0.0
        # mtime 热加载:任一文件变化即整体重扫(目录小,代价可忽略)
        defs_dir = default_definitions_dir()
        try:
            newest = max((f.stat().st_mtime for f in defs_dir.rglob("SKILL.md")),
                         default=0.0)
        except OSError:
            newest = 0.0
        if newest > _scanned_at:
            _definitions = _scan_dir(defs_dir)
            _scanned_at = newest
        return dict(_definitions)


def reload_definitions() -> dict[str, SkillDefinition]:
    """强制重载(管理台/CI 用)。"""
    return load_all_definitions(force=True)


# ── handler 解析与注册 ──────────────────────────────────────────


def _resolve_handler(dotted: str):
    """'user_segment.UserSegmentStatsSkill' → 类。模块优先从 skills 包解析。"""
    mod_name, _, cls_name = dotted.rpartition(".")
    if not mod_name or not cls_name:
        raise SkillDefinitionError(f"handler 格式错误: {dotted}(需 '模块.类')")
    try:
        mod = __import__(f"skills.{mod_name}", fromlist=[cls_name])
    except ImportError:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
        except ImportError as e:
            raise SkillDefinitionError(f"handler 模块不可导入: {dotted} ({e})")
    cls = getattr(mod, cls_name, None)
    if cls is None or not isinstance(cls, type):
        raise SkillDefinitionError(f"handler 类不存在: {dotted}")
    return cls


def register_from_definitions(
    defs: dict[str, SkillDefinition] | None = None,
) -> list[str]:
    """把 kind=tool 的定义实例化并注册进 SkillRegistry,返回已注册名字列表。

    frontmatter 覆盖类属性(description/group/version/tags),定义即行为;
    kind=instruction 的定义不注册(仅留在定义库供选择器披露)。
    """
    from skills import SkillRegistry

    defs = defs or load_all_definitions()
    registered: list[str] = []
    for name, defn in sorted(defs.items()):
        if defn.kind != "tool":
            continue
        try:
            cls = _resolve_handler(defn.handler)
            skill = cls()
            # 声明式覆盖:SKILL.md 是唯一事实来源(含 name 本身)
            if skill.name != defn.name:
                logger.warning("skill_name_mismatch", extra={
                    "definition_name": defn.name, "class_name": skill.name,
                })
            skill.name = defn.name
            skill.description = defn.description or skill.description
            skill.group = defn.group
            skill.kind = defn.kind
            skill.version = defn.version
            skill.tags = list(defn.tags)
            skill.definition = defn
            SkillRegistry.register(skill)
            registered.append(name)
            logger.info("skill_registered_from_file", extra={
                "skill_name": name, "group": defn.group, "version": defn.version,
            })
        except Exception as e:
            logger.error("skill_register_failed", extra={
                "skill_name": name, "error": str(e),
            })
    return registered
