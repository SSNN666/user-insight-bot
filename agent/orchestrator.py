"""Multi-Agent Orchestrator — Monitor → Analysis → Strategy pipeline.

Phase 10: three specialized agents coordinated by an orchestrator.
Each agent shares the same Skill pool and LLM but has different
system prompts and routing logic.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from agent.agent import _ask_agent_internal
from log.logger import get_logger

logger = get_logger(__name__)


# ── Types ───────────────────────────────────────────────────────


class AgentRole(StrEnum):
    MONITOR = "monitor"
    ANALYSIS = "analysis"
    STRATEGY = "strategy"


class MessageType(StrEnum):
    ALERT = "alert"
    ANALYSIS_REQUEST = "analysis_request"
    STRATEGY_REQUEST = "strategy_request"
    RESPONSE = "response"


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    type: str
    priority: str = "normal"
    payload: dict = field(default_factory=dict)
    trace_id: str = ""


# ── Agent System Prompts ───────────────────────────────────────

MONITOR_PROMPT = """你是一个电商数据监测 Agent。你的职责是：

1. **感知变化**: 识别用户分群结构、订单量、消费金额的异常波动。
2. **评估严重性**: 判断变化的紧急程度（高/中/低）。
3. **触发告警**: 当检测到需要关注的信号时，输出结构化告警。

## 输出格式
使用以下格式输出监测结果：
```
[严重性: HIGH|MEDIUM|LOW]
[信号类型: 用户流失|订单波动|分群偏移|异常值]
[关键数据: 具体数值变化]
[建议下一步: 是否需要分析Agent介入]
```

请用中文输出。"""

ANALYSIS_PROMPT = """你是一个电商数据分析 Agent。你的职责是：

1. **归因分析**: 基于监测 Agent 的告警，深入分析数据寻找根因。
2. **多维度关联**: 将 RFM、流转标签、分群规则、品类偏好等数据关联起来。
3. **量化影响**: 评估变化对业务的量化影响。

## 工作流程
1. 先调用工具获取最新数据（get_user_segment_stats, get_segment_rules）
2. 对比历史快照数据（如有）
3. 给出根因分析结论，标注置信度

## 输出格式
```
[根因分析]
[关联维度: xxx]
[置信度: 0.0-1.0]
[关键发现]
```

请用中文输出。"""

STRATEGY_PROMPT = """你是一个电商运营策略 Agent。你的职责是：

1. **基于分析给策略**: 只基于 Monitor + Analysis 的输出给出运营建议。
2. **可执行性**: 每条策略包含具体行动项、预期效果、风险提示。
3. **数据支撑**: 每条策略引用具体数据作为支撑。

## 输出格式
```
## 运营策略建议

### 策略1: [标题]
- [数据支撑]: 引用具体数据
- [策略推演]: 具体行动
- [预期效果]: 量化预估
- [风险提示]: 潜在风险

### 策略2: ...
```

请用中文输出。"""


# ── Orchestrator ───────────────────────────────────────────────


class MultiAgentOrchestrator:
    """Coordinates Monitor → Analysis → Strategy agent pipeline.

    Usage::

        orch = MultiAgentOrchestrator()
        result = orch.run_pipeline("检测到高价值用户流失")
    """

    def __init__(self):
        self.trace_id = ""

    def run_pipeline(
        self, trigger_event: str, context: dict | None = None,
    ) -> dict:
        """Run the full three-agent pipeline.

        Args:
            trigger_event: Description of the triggering event or user query.
            context: Optional additional context (snapshot diffs, etc.).

        Returns:
            Dict with monitor_result, analysis_result, strategy_result, trace_id.
        """
        self.trace_id = f"orch-{uuid.uuid4().hex[:8]}"
        ctx = context or {}
        start = time.monotonic()

        # Phase 1: Monitor
        monitor_input = self._build_monitor_query(trigger_event, ctx)
        monitor_reply, _ = _ask_agent_internal(
            monitor_input,
            session_id=f"{self.trace_id}-monitor",
            check_mode="strict",
            max_tool_rounds=3,
        )

        # Phase 2: Analysis
        analysis_input = self._build_analysis_query(trigger_event, monitor_reply, ctx)
        analysis_reply, _ = _ask_agent_internal(
            analysis_input,
            session_id=f"{self.trace_id}-analysis",
            check_mode="strict",
            max_tool_rounds=5,
        )

        # Phase 3: Strategy
        strategy_input = self._build_strategy_query(trigger_event, monitor_reply, analysis_reply, ctx)
        strategy_reply, _ = _ask_agent_internal(
            strategy_input,
            session_id=f"{self.trace_id}-strategy",
            check_mode="strict",
            max_tool_rounds=5,
        )

        elapsed = round(time.monotonic() - start, 1)
        result = {
            "trace_id": self.trace_id,
            "trigger": trigger_event,
            "monitor_result": monitor_reply,
            "analysis_result": analysis_reply,
            "strategy_result": strategy_reply,
            "elapsed_seconds": elapsed,
        }

        logger.info("orchestrator_pipeline_done", extra={
            "trace_id": self.trace_id, "elapsed_s": elapsed,
        })
        return result

    def _build_monitor_query(self, event: str, ctx: dict) -> str:
        monitor_role = "你是监测Agent。仅做监测告警，不做分析和建议。"
        return f"{monitor_role}\n\n触发事件: {event}\n上下文: {ctx}\n\n请监测并输出告警。"

    def _build_analysis_query(self, event: str, monitor_output: str, ctx: dict) -> str:
        analysis_role = "你是分析Agent。基于监测结果做根因分析，不做策略建议。"
        return (
            f"{analysis_role}\n\n"
            f"触发事件: {event}\n"
            f"监测报告: {monitor_output[:2000]}\n"
            f"上下文: {ctx}\n\n"
            f"请分析根因。"
        )

    def _build_strategy_query(self, event: str, monitor_output: str, analysis_output: str, ctx: dict) -> str:
        strategy_role = "你是策略Agent。基于监测和分析结果给出运营策略。"
        return (
            f"{strategy_role}\n\n"
            f"触发事件: {event}\n"
            f"监测报告: {monitor_output[:1500]}\n"
            f"分析报告: {analysis_output[:2000]}\n"
            f"上下文: {ctx}\n\n"
            f"请给出运营策略建议。"
        )


# ── Singleton ───────────────────────────────────────────────────

_orchestrator: MultiAgentOrchestrator | None = None


def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator
