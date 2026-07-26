"""LangChain tools — thin compatibility layer over the Skill framework.

All actual logic lives in ``skills/``. This module:
1. Registers built-in skills on first import.
2. Exposes ``tools`` list consumed by ``agent/agent.py``.
"""

from skills import SkillRegistry
from skills.user_segment import register_all_skills

# ── One-time registration ───────────────────────────────────────
register_all_skills()

# ── Public tool list for agent ───────────────────────────────────
tools = SkillRegistry.get_langchain_tools()
