"""Agentic RAG agent — custom StateGraph with preprocess / reflect / fact-check.

Graph topology (Phase 2):

    START → preprocess → llm_decide → [tools → reflect → llm_decide]
                              │                                    │
                              └──→ respond → fact_check → END      │
                                          ↑            │           │
                                          └────────────┘ (strict retry)

Key features:
- LLM autonomously decides whether to call tools (Agentic RAG).
- Query preprocessing enriches the state with intent + entities.
- Self-reflection validates tool results before answering.
- Three-layer rule-based fact-check against SkillResult.data ground truth.
- Dual-mode enforcement: relaxed (warn) / strict (block + retry).
- Three scenario-specific prompt templates with keyword routing.
- Mandatory Markdown output format.
- Dual-layer memory: LangGraph checkpoints (chat) + BusinessMemory (task).
- Unified LLM adapter (llm/bridge.py): timeout / 429 / quota / context-overflow
  handling + multi-provider fallback chain + token usage collection.
- Real per-node timings recorded into AgentState.node_timings (no estimates).
- SSE streaming: nodes emit node_start / tool_call / tool_result events via
  get_stream_writer; _ask_agent_stream consumes astream updates+messages+custom.
"""

import time
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from config.settings import get_settings
from agent.system_prompt import SYSTEM_PROMPT
from agent.prompts import (
    select_prompt_template,
    PROMPT_DATA_STATS,
    enforce_markdown,
)
from agent.preprocessor import preprocess_query, PreprocessResult
from agent.reflector import reflect, ReflectResult
from agent.memory import get_business_memory, clear_business_memory
from agent.session_store import LazyCheckpointSaver
from agent.chart_extract import build_charts_from_skills
from llm.bridge import get_fallback_llm, start_usage_collection
from errors.exceptions import APIError
from log.logger import get_logger

logger = get_logger(__name__)


def _seg_names_safe() -> dict:
    """图表标注用分群业务名(失败返回空 dict,不阻塞主链路)。"""
    try:
        from pipeline.segment_naming import get_segment_names
        return get_segment_names()
    except Exception:
        return {}


# ── Extended State ──────────────────────────────────────────────


class AgentState(MessagesState):
    """State for the Agentic RAG graph (Phase 2)."""
    preprocess_result: PreprocessResult | None
    skill_results: list[Any]             # list[SkillResult] — captured directly
    reflect_result: ReflectResult | None
    fact_check_result: Any | None        # FactCheckResult
    check_mode: str                       # "relaxed" | "strict"
    max_tool_rounds: int | None           # Phase 4: per-invocation override
    iteration: int
    fact_check_retries: int               # strict 模式事实核查重试次数(独立于 reflect 轮数)
    should_continue: bool
    node_timings: dict[str, float]        # node → real elapsed ms (measured, not estimated)
    system_override: str | None           # 角色级 system prompt 覆盖(orchestrator 三角色使用)
    force_analysis: bool                   # 强制分析模式(管理台):购物意图不回退购物助手话术
    tools_called_names: list[str]         # 实际调用的 Skill 名(eval/可观测性读取真实值)
    conversation_summary: str | None      # 长对话历史摘要(超阈值时由 preprocess 计算)


# ── Helpers ─────────────────────────────────────────────────────


def _build_llm(role: str = "decide"):
    """LLM via the unified adapter (timeout / fallback chain / token stats)."""
    return get_fallback_llm(role)


def _build_llm_with_tools(tool_names: list[str] | None = None):
    """LLM that knows about available skills (via bind_tools).

    Args:
        tool_names: Optional list of tool names to expose. If None, all tools.
                    Scoping tools per intent prevents the LLM from calling
                    inappropriate tools (e.g. segment tools during shopping).
    """
    return get_fallback_llm("decide", tool_names=tool_names)


def _emit(event: str, **kwargs) -> None:
    """Send a streaming event via get_stream_writer (no-op outside astream custom mode).

    Non-streaming callers (watcher engine, plain /ask) never set a stream writer,
    so this silently does nothing for them.
    """
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"event": event, **kwargs})
    except Exception:
        pass


def _finish_node_timing(state: AgentState, node: str, t0: float) -> dict:
    """Accumulate real per-node elapsed time into state (revisits add up)."""
    elapsed = round((time.monotonic() - t0) * 1000, 1)
    timings = dict(state.get("node_timings") or {})
    timings[node] = round(timings.get(node, 0) + elapsed, 1)
    return timings


SUMMARY_PROMPT = """请把以下对话历史压缩为要点摘要(不超过150字):
- 保留:用户关注的数据主题、已确认的结论、用户的偏好信息
- 丢弃:客套话、重复内容、被修正过的错误结论
只输出摘要正文,不要标题或解释。"""


def _summarize_conversation(history: list, llm) -> str | None:
    """长对话历史 → 要点摘要(LLM 调用,失败返回 None 降级为不压缩)。"""
    try:
        lines = []
        for m in history:
            if getattr(m, "type", None) == "human":
                lines.append(f"用户: {getattr(m, 'content', '')[:200]}")
            elif getattr(m, "type", None) == "ai":
                lines.append(f"AI: {getattr(m, 'content', '')[:300]}")
        if not lines:
            return None
        resp = llm.invoke([
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": "\n".join(lines[-40:])},
        ])
        text = (resp.content or "").strip()
        return text or None
    except Exception as exc:
        logger.warning("conversation_summary_failed", extra={"error": str(exc)[:200]})
        return None


def extract_products(skill_results: list) -> list[dict]:
    """从 SkillResult 中提取商品列表(供前端渲染商品卡片/一键加购)。

    商品类 Skill(search_products / get_personal_recommendations)的 data 都是
    [{"product_id","product_name","category","price"}, ...] 结构。
    """
    for sr in skill_results or []:
        data = getattr(sr, "data", None)
        if isinstance(data, list) and data and isinstance(data[0], dict) \
                and "product_id" in data[0] and "product_name" in data[0]:
            return [
                {
                    "product_id": d.get("product_id"),
                    "product_name": d.get("product_name", ""),
                    "category": d.get("category", ""),
                    "price": d.get("price", 0),
                }
                for d in data[:5]
            ]
    return []


def extract_ai_response(messages: list) -> str:
    """Find the last AI message by *role/type*, not array position."""
    ai_messages = [
        m for m in messages
        if getattr(m, 'type', None) == 'ai'
        and getattr(m, 'content', None)
    ]
    if not ai_messages:
        raise APIError(
            "No AI response found in messages",
            context={"message_count": len(messages)},
        )
    return ai_messages[-1].content


def extract_last_ai_text(update: dict) -> str:
    """Extract the last AI text from a state-update messages delta (may be empty)."""
    for m in reversed(update.get("messages", []) or []):
        if getattr(m, 'type', None) == 'ai' and getattr(m, 'content', None):
            return m.content
    return ""


# ── Graph Nodes ─────────────────────────────────────────────────


def _preprocess_node(state: AgentState) -> dict:
    """Classify intent, extract entities, decompose query."""
    t0 = time.monotonic()
    _emit("node_start", node="preprocess")
    settings = get_settings()
    if not settings.PREPROCESS_ENABLED:
        return {
            "preprocess_result": None, "iteration": 0,
            "skill_results": [], "should_continue": True,
            "tools_called_names": [],
            "node_timings": _finish_node_timing(state, "preprocess", t0),
        }

    user_msgs = [m for m in state["messages"]
                 if getattr(m, 'type', None) == 'human']
    query = user_msgs[-1].content if user_msgs else ""

    result = preprocess_query(query, _build_llm("preprocess"))

    # ── 长对话摘要:消息超阈值 → 压缩旧历史(保持上下文有界) ──
    summary = state.get("conversation_summary")
    if settings.CONVERSATION_SUMMARY_ENABLED:
        msgs = state["messages"]
        if len(msgs) > settings.CONVERSATION_SUMMARY_THRESHOLD:
            history = msgs[:-settings.CONVERSATION_KEEP_RECENT]
            summary = _summarize_conversation(history, _build_llm("preprocess"))
            _emit("node_end_detail", node="preprocess", summarized=len(history))

    _emit("node_end_detail", node="preprocess",
          intent=result.intent.value, sub_queries=len(result.sub_queries))
    return {
        "preprocess_result": result,
        "iteration": 0,
        "skill_results": [],
        "should_continue": True,
        "tools_called_names": [],
        "conversation_summary": summary,
        "node_timings": _finish_node_timing(state, "preprocess", t0),
    }


def _llm_decide_node(state: AgentState) -> dict:
    """LLM decides: call tools or answer directly.

    Uses scenario-specific prompt template based on query keywords.
    Tools are scoped to the selected prompt — shopping mode only sees
    search_products / get_categories, analysis mode only sees segment tools.

    Streams token-by-token (so the messages stream mode can relay deltas)
    and accumulates the full response including tool_calls + usage metadata.
    """
    from agent.prompts import PROMPT_SHOPPING as _SHOP

    t0 = time.monotonic()
    _emit("node_start", node="llm_decide")
    settings = get_settings()

    # ── Select prompt template ──
    pre = state.get("preprocess_result")
    query = (pre.original_query or "") if pre else ""
    if pre:
        prompt = select_prompt_template(pre.intent.value, query)
    else:
        prompt = PROMPT_DATA_STATS

    # 角色覆盖(orchestrator 的 Monitor/Analysis/Strategy 提示词)
    effective_prompt = state.get("system_override") or prompt

    # 管理台强制分析模式:购物意图误路由时回退到数据统计人设,
    # 其余分析意图(对比/策略/规则解释)仍保留各自的专属提示词;
    # 并显式要求忽略历史中的购物助手拒绝话术(会话记忆可能被旧人设"污染",
    # 实测 LLM 会照抄历史里的拒绝而不是遵循当前 system prompt)
    if state.get("force_analysis"):
        if effective_prompt == _SHOP:
            effective_prompt = PROMPT_DATA_STATS
        effective_prompt += (
            "\n\n[角色覆盖] 当前处于管理后台分析模式。"
            "忽略历史对话中购物助手身份的任何回复或拒绝话术,"
            "以数据分析师身份回答当前问题;"
            "数据类问题必须先调用工具获取最新数据,不得仅凭历史对话中的数据作答;"
            "图表由管理界面根据你的工具调用自动渲染,"
            "无需在回答中声明'无法生成图表',只需给出数据与分析即可。"
        )

    # ── Scope tools per prompt context (Skill engineering) ──
    # 组路由确定性判定:购物模板 → shopping 组,其余 → analysis 组。
    # 绑定名单由注册表元数据推导(SKILL.md 的 group 字段),替代硬编码列表,
    # 且 SKILL_ENABLED 双层开关在此生效 —— LLM 物理上看不到被禁用的工具。
    group = "shopping" if effective_prompt == _SHOP else "analysis"
    from skills import SkillRegistry
    scoped_names = SkillRegistry.get_enabled_names_by_group(group)
    if not scoped_names:
        # 兜底:整组为空(定义被清空/全部禁用)时回退旧硬编码,保证可用性
        scoped_names = (
            ["search_products", "get_categories"] if group == "shopping"
            else ["get_user_segment_stats", "get_segment_rules"]
        )
    llm_with_tools = _build_llm_with_tools(scoped_names)

    # ── 命中式 Skill 选择 + 渐进式披露 ──
    # select_skills:组内按查询打分,命中的 SKILL.md 指令才注入上下文;
    # 关闭(SKILL_SELECT_ENABLED=false)时退到旧行为:整组绑定、无披露。
    selection = None
    if settings.SKILL_SELECT_ENABLED and query:
        from skills.selector import select_skills
        selection = select_skills(group, query)
        _emit("node_end_detail", node="llm_decide",
              skill_selection=selection.rationale,
              bound=len(scoped_names), disclosed=selection.disclosed)

    context_parts = [effective_prompt]

    # Inject preprocess result
    if pre and pre.intent.value != "general":
        context_parts.append(f"\n[查询分析] {pre.format_for_context()}")

    # Progressive disclosure: 命中的 SKILL.md 指令注入(选择器产物,
    # 比通用输出要求更具体,LLM 优先遵循)
    if selection is not None and selection.disclosed:
        from skills.selector import disclosure_block
        block = disclosure_block(selection)
        if block:
            context_parts.append(f"\n{block}")

    # Inject 分群业务命名(运营说"高价值用户",不说"分群2")
    try:
        from pipeline.segment_naming import get_segment_names
        names = get_segment_names()
        if names:
            context_parts.append(
                "\n[分群业务命名] "
                + ", ".join(f"分群{k}={v}" for k, v in sorted(names.items()))
            )
    except Exception:
        pass

    # Inject business memory
    session_id = state.get("session_id", "default")
    biz_mem = get_business_memory(session_id)
    biz_context = biz_mem.format_for_context()
    if biz_context:
        context_parts.append(f"\n[历史分析上下文]\n{biz_context}")

    # Phase 7: Flywheel — retrieve relevant high-quality Q&A samples
    try:
        from flywheel.retriever import retrieve_relevant_samples
        query_str = pre.original_query if pre else ""
        if not query_str:
            user_msgs = [m for m in state.get("messages", [])
                         if getattr(m, 'type', None) == 'human']
            query_str = user_msgs[-1].content if user_msgs else ""
        relevant = retrieve_relevant_samples(query_str, top_k=3)
        if relevant:
            samples_text = "\n---\n".join(
                f"Q: {s['question'][:300]}\nA: {s['reply'][:500]}"
                for s in relevant
            )
            context_parts.append(
                f"\n[历史优质回答参考（可借鉴风格和结构，但需基于当前数据重新分析）]\n{samples_text}"
            )
    except ImportError:
        logger.debug("flywheel_not_available")
    except Exception as exc:
        logger.warning("flywheel_retrieval_failed", extra={
            "error": str(exc)[:200],
        })

    enriched_system = "\n".join(context_parts)

    # ── Hard tool-call guard ──
    # Only active in analysis mode (not shopping). Shopping mode relies
    # on the natural LLM flow — users can just chat without tools.
    skill_results = state.get("skill_results", [])
    if settings.TOOL_CALL_GUARD_ENABLED and not skill_results and effective_prompt != _SHOP:
        user_msgs = [m for m in state.get("messages", [])
                     if getattr(m, 'type', None) == 'human']
        query = user_msgs[-1].content if user_msgs else ""
        if any(kw in query for kw in settings.TOOL_CALL_GUARD_KEYWORDS):
            guard_names = ", ".join(scoped_names)
            guard = (
                f"\n\n🚫 **系统拦截: 你尚未调用任何工具。** "
                f"你必须立即调用以下工具之一获取实际数据: {guard_names}。"
                f"在工具返回结果之前，禁止输出任何表格或数据。"
                f"如果你输出了编造的数据，回答将被丢弃。"
            )
            enriched_system += guard

    # Build messages: system + (可选)历史摘要 + 最近原文
    # 长对话时用摘要替代旧历史,上下文保持有界(摘要由 preprocess 计算)
    messages = [{"role": "system", "content": enriched_system}]
    summary = state.get("conversation_summary")
    if summary:
        messages.append({"role": "system", "content": f"[对话历史摘要] {summary}"})
        recent = state["messages"][-settings.CONVERSATION_KEEP_RECENT:]
        messages.extend(recent)
    else:
        messages.extend(state["messages"])

    # ── Stream & accumulate (token deltas + final tool_calls + usage) ──
    text_parts: list[str] = []
    final_tool_calls: list[dict] = []
    usage_meta = None
    resp_meta: dict = {}
    for chunk in llm_with_tools.stream(messages):
        if chunk.content:
            text_parts.append(chunk.content)
        if getattr(chunk, "usage_metadata", None):
            usage_meta = dict(chunk.usage_metadata)
        if getattr(chunk, "response_metadata", None):
            resp_meta.update(chunk.response_metadata)
        if getattr(chunk, "tool_calls", None):
            final_tool_calls = chunk.tool_calls

    response = AIMessage(
        content="".join(text_parts),
        tool_calls=final_tool_calls,
        usage_metadata=usage_meta,
        response_metadata=resp_meta,
    )

    # ── Tool-call text-pattern → native tool_call bridge ──
    # Some Ollama models output text like "get_user_segment_stats"
    # instead of using the native tool_calls API. Parse & convert.
    if not getattr(response, 'tool_calls', None):
        content = getattr(response, 'content', '') or ''
        import re as _re
        found_tools = [tn for tn in scoped_names if tn in content]
        if found_tools:
            # Create synthetic tool_calls from text match
            synthetic_calls = [
                {"name": tn, "args": {}, "id": f"synth-{i}"}
                for i, tn in enumerate(found_tools[:3])
            ]
            response = AIMessage(
                content="",
                tool_calls=synthetic_calls,
            )

    return {"messages": [response],
            "node_timings": _finish_node_timing(state, "llm_decide", t0)}


def _tools_node(state: AgentState) -> dict:
    """Execute tool calls directly via SkillRegistry, capturing SkillResult objects.

    This replaces ToolNode so we can store actual SkillResult.data
    as ground truth for the fact-checker.
    """
    t0 = time.monotonic()
    _emit("node_start", node="tools")

    last_msg = state["messages"][-1] if state["messages"] else None
    skill_results: list[Any] = list(state.get("skill_results", []))
    called_names: list[str] = list(state.get("tools_called_names", []))
    tool_messages: list[ToolMessage] = []

    if last_msg and hasattr(last_msg, "tool_calls"):
        for i, tc in enumerate(last_msg.tool_calls):
            skill_name = tc.get("name", "unknown")
            skill_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")
            called_names.append(skill_name)

            _emit("tool_call", name=skill_name, args=skill_args, index=i)
            skill = SkillRegistry_get(skill_name)
            t_skill = time.monotonic()
            if skill:
                try:
                    result = skill.execute(**skill_args)
                    skill_results.append(result)
                    tool_messages.append(ToolMessage(
                        content=result.to_context_string(),
                        tool_call_id=tool_call_id,
                        name=skill_name,
                    ))
                    _emit("tool_result", name=skill_name,
                          status=result.status.value,
                          elapsed_ms=round((time.monotonic() - t_skill) * 1000, 1),
                          summary=result.summary[:120])
                except Exception as exc:
                    from skills.base import SkillResult as SR, SkillStatus
                    err_result = SR(
                        status=SkillStatus.ERROR,
                        error=str(exc),
                        confidence=0.0,
                    )
                    skill_results.append(err_result)
                    tool_messages.append(ToolMessage(
                        content=err_result.to_context_string(),
                        tool_call_id=tool_call_id,
                        name=skill_name,
                    ))
                    _emit("tool_result", name=skill_name, status="error",
                          elapsed_ms=round((time.monotonic() - t_skill) * 1000, 1),
                          summary=str(exc)[:120])
            else:
                tool_messages.append(ToolMessage(
                    content=f"Skill '{skill_name}' not found.",
                    tool_call_id=tool_call_id,
                    name=skill_name,
                ))
                _emit("tool_result", name=skill_name, status="error",
                      elapsed_ms=round((time.monotonic() - t_skill) * 1000, 1),
                      summary="skill not found")

    # Args JSON parse failures (invalid_tool_calls from the bridge)
    if last_msg and hasattr(last_msg, "invalid_tool_calls") and last_msg.invalid_tool_calls:
        for itc in last_msg.invalid_tool_calls:
            tool_messages.append(ToolMessage(
                content=f"工具参数解析失败: {itc.get('error', 'invalid arguments')}",
                tool_call_id=itc.get("id", ""),
                name=itc.get("name", "unknown"),
            ))

    # Update business memory
    session_id = state.get("session_id", "default")
    biz_mem = get_business_memory(session_id)
    if last_msg and hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            biz_mem.add_skill_call(tc.get("name", "unknown"))
    biz_mem.analysis_state = "tools_executed"

    return {"messages": tool_messages, "skill_results": skill_results,
            "tools_called_names": called_names,
            "node_timings": _finish_node_timing(state, "tools", t0)}


def _reflect_node(state: AgentState) -> dict:
    """Validate tool results; decide whether to re-query."""
    t0 = time.monotonic()
    _emit("node_start", node="reflect")
    settings = get_settings()
    iteration = state.get("iteration", 0) + 1

    skill_results = state.get("skill_results", [])

    user_msgs = [m for m in state["messages"]
                 if getattr(m, 'type', None) == 'human']
    original_query = user_msgs[-1].content if user_msgs else ""

    reflect_result = reflect(skill_results, original_query, _build_llm("reflect"))
    _emit("node_end_detail", node="reflect",
          is_complete=reflect_result.is_complete, confidence=reflect_result.confidence)

    should_continue = (
        not reflect_result.is_complete
        and iteration < state.get("max_tool_rounds", settings.MAX_REFLECTION_ROUNDS)
        and reflect_result.suggestion != ""
    )

    if not reflect_result.is_complete:
        hint = (
            f"[自反思] 数据不完整。缺失: {reflect_result.missing_info}. "
            f"建议: {reflect_result.suggestion}"
        )
        return {
            "reflect_result": reflect_result,
            "iteration": iteration,
            "should_continue": should_continue,
            "messages": [{"role": "system", "content": hint}],
            "node_timings": _finish_node_timing(state, "reflect", t0),
        }

    return {
        "reflect_result": reflect_result,
        "iteration": iteration,
        "should_continue": False,
        "node_timings": _finish_node_timing(state, "reflect", t0),
    }


def _respond_node(state: AgentState) -> dict:
    """Ensure the final response is valid Markdown."""
    t0 = time.monotonic()
    _emit("node_start", node="respond")
    ai_msgs = [
        m for m in state["messages"]
        if getattr(m, 'type', None) == 'ai'
        and getattr(m, 'content', None)
    ]
    if not ai_msgs:
        return {"node_timings": _finish_node_timing(state, "respond", t0)}

    last_ai = ai_msgs[-1]
    formatted = enforce_markdown(last_ai.content)

    if formatted != last_ai.content:
        amended_msg = last_ai.model_copy(update={"content": formatted})
        new_messages = list(state["messages"])
        for i in range(len(new_messages) - 1, -1, -1):
            if getattr(new_messages[i], 'type', None) == 'ai':
                new_messages[i] = amended_msg
                break
        return {"messages": new_messages,
                "node_timings": _finish_node_timing(state, "respond", t0)}

    return {"node_timings": _finish_node_timing(state, "respond", t0)}


def _fact_check_node(state: AgentState) -> dict:
    """Run three-layer fact-check on the last AI response."""
    from agent.fact_checker import run_fact_check

    t0 = time.monotonic()
    _emit("node_start", node="fact_check")
    settings = get_settings()
    check_mode = state.get("check_mode", settings.FACT_CHECK_MODE)

    ai_msgs = [
        m for m in state["messages"]
        if getattr(m, 'type', None) == 'ai'
        and getattr(m, 'content', None)
    ]
    if not ai_msgs:
        return {"fact_check_result": None,
                "node_timings": _finish_node_timing(state, "fact_check", t0)}

    llm_response = ai_msgs[-1].content
    skill_results = state.get("skill_results", [])

    result = run_fact_check(llm_response, skill_results, check_mode)

    if not result.passed and check_mode == "strict":
        correction_text = result.format_corrections_for_llm()
        return {
            "fact_check_result": result,
            "messages": [{"role": "system", "content": correction_text}],
            # 独立计数:LLM 不调工具直接答题时 reflect 不会执行,iteration 恒 0,
            # 若共用它会导致重试无上限(依赖 recursion_limit 兜底抛 RecursionError)
            "fact_check_retries": state.get("fact_check_retries", 0) + 1,
            "node_timings": _finish_node_timing(state, "fact_check", t0),
        }
    elif not result.passed and check_mode == "relaxed":
        warning_text = result.format_violations_for_user()
        last_ai = ai_msgs[-1]
        amended_content = last_ai.content + warning_text
        amended_msg = last_ai.model_copy(update={"content": amended_content})
        new_messages = list(state["messages"])
        for i in range(len(new_messages) - 1, -1, -1):
            if getattr(new_messages[i], 'type', None) == 'ai':
                new_messages[i] = amended_msg
                break
        return {
            "fact_check_result": result,
            "messages": new_messages,
            "node_timings": _finish_node_timing(state, "fact_check", t0),
        }

    return {"fact_check_result": result,
            "node_timings": _finish_node_timing(state, "fact_check", t0)}


# ── Conditional Edges ───────────────────────────────────────────


def _reflect_condition(state: AgentState) -> Literal["llm_decide", "respond"]:
    """After reflection: decide next step.

    If data is still incomplete and we have retries left → loop to llm_decide.
    If data is complete but the last AI message was a tool_call → must go to
    llm_decide so the LLM can generate a text response from the tool results.
    Only skip to respond if the LLM already answered without tools.
    """
    if state.get("should_continue", False):
        logger.info("reflect_loop", extra={
            "iteration": state.get("iteration", 0),
        })
        return "llm_decide"

    # Check if last AI message has tool_calls — if so, must go back to llm_decide
    last_ai = None
    for m in reversed(state.get("messages", [])):
        if getattr(m, 'type', None) == 'ai':
            last_ai = m
            break
    if last_ai and hasattr(last_ai, 'tool_calls') and last_ai.tool_calls:
        return "llm_decide"

    return "respond"


def _fact_check_condition(state: AgentState) -> Literal["llm_decide", "__end__"]:
    """After fact-check: loop back in strict mode if violations found."""
    settings = get_settings()
    fcr = state.get("fact_check_result")
    check_mode = state.get("check_mode", settings.FACT_CHECK_MODE)

    if (
        fcr and not fcr.passed
        and check_mode == "strict"
        and state.get("fact_check_retries", 0) < settings.FACT_CHECK_MAX_RETRIES
    ):
        logger.info("fact_check_retry", extra={
            "violations": len(fcr.violations),
            "retries": state.get("fact_check_retries", 0),
        })
        return "llm_decide"
    return "__end__"


# ── Build Graph ─────────────────────────────────────────────────


def _build_graph(checkpointer=None):
    """Construct the Agentic RAG StateGraph (Phase 2).

    checkpointer: 默认惰性 JSON 持久化(会话重启不丢);测试可注入自定义 saver。
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("preprocess", _preprocess_node)
    workflow.add_node("llm_decide", _llm_decide_node)
    workflow.add_node("tools", _tools_node)
    workflow.add_node("reflect", _reflect_node)
    workflow.add_node("respond", _respond_node)
    workflow.add_node("fact_check", _fact_check_node)

    # Edges
    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "llm_decide")

    # llm_decide → tools (if tool_calls) or respond (if done)
    workflow.add_conditional_edges(
        "llm_decide",
        tools_condition,
        {"tools": "tools", "__end__": "respond"},
    )

    # tools → reflect
    workflow.add_edge("tools", "reflect")

    # reflect → llm_decide (re-query) or respond (done)
    workflow.add_conditional_edges(
        "reflect",
        _reflect_condition,
        {"llm_decide": "llm_decide", "respond": "respond"},
    )

    # respond → fact_check → END (or loop back in strict mode)
    workflow.add_edge("respond", "fact_check")
    workflow.add_conditional_edges(
        "fact_check",
        _fact_check_condition,
        {"llm_decide": "llm_decide", "__end__": END},
    )

    # Compile with checkpoint for session memory
    # 默认惰性 JSON 持久化:进程重启后同一 thread_id 的多轮上下文自动恢复
    memory_saver = checkpointer if checkpointer is not None else LazyCheckpointSaver()
    return workflow.compile(checkpointer=memory_saver)


# ── Module-level graph ──────────────────────────────────────────

_agent_graph = _build_graph()


def SkillRegistry_get(name: str):
    from skills import SkillRegistry
    return SkillRegistry.get(name)


# ── Public API ──────────────────────────────────────────────────


def _ask_agent_internal(
    question: str,
    session_id: str = "default",
    check_mode: str | None = None,
    max_tool_rounds: int | None = None,
    system_override: str | None = None,
    force_analysis: bool = False,
) -> tuple[str, int, list[dict], list[dict]]:
    """Invoke the graph and return (reply_text, iteration_count, products, charts).

    products: 商品类 Skill 返回的商品列表(前端商品卡片/一键加购),无商品时为空。
    charts: 分析类 Skill 数据透出的图表协议(前端 SVG / Gradio matplotlib 渲染)。
    Used by the watcher engine to track tool-call rounds. ``system_override``
    replaces the scenario prompt (orchestrator 的 Monitor/Analysis/Strategy 角色)。
    ``force_analysis``: 管理台模式,购物意图误路由时回退数据统计人设。
    """
    start = time.monotonic()
    settings = get_settings()
    mode = check_mode or settings.FACT_CHECK_MODE

    # ── Agent Trace (Phase: observability) ──
    from agent.tracer import start_trace, record_span, finish_trace
    trace = start_trace(session_id, question)

    # Per-run LLM usage collection (ContextVar — preprocess/reflect calls
    # don't enter the messages stream, so they're collected via the bridge).
    usage_bucket = start_usage_collection()

    try:
        result = _agent_graph.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "session_id": session_id,
                "check_mode": mode,
                "max_tool_rounds": max_tool_rounds or settings.MAX_REFLECTION_ROUNDS,
                "system_override": system_override,
                "force_analysis": force_analysis,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        total_elapsed = round((time.monotonic() - start) * 1000)

        # ── Record trace spans from state (real measured durations) ──
        timings: dict[str, float] = result.get("node_timings") or {}

        def _role_tokens(role: str) -> tuple[int, str]:
            usages = usage_bucket.get(role, [])
            total = sum(u.total_tokens for u in usages)
            provider = usages[-1].provider if usages else ""
            return total, provider

        # preprocess
        pre = result.get("preprocess_result")
        if pre:
            p_tokens, _ = _role_tokens("preprocess")
            record_span(session_id, "preprocess", timings.get("preprocess", 0),
                        tokens_used=p_tokens,
                        input_summary=question[:100],
                        output_summary=f"意图: {pre.intent.value} | 子任务: {pre.sub_queries if hasattr(pre,'sub_queries') else '无'}",
                        metadata={"intent": pre.intent.value, "entities": pre.entities})

        # llm_decide + tools
        skill_results = result.get("skill_results", [])
        tools_called = []
        for sr in skill_results:
            name = getattr(sr, 'summary', '')[:100] or str(sr)[:100]
            tools_called.append(name)
        tool_names = result.get("tools_called_names", [])   # 实际 Skill 名(eval/可观测性)
        d_tokens, d_provider = _role_tokens("decide")
        if tools_called:
            record_span(session_id, "llm_decide", timings.get("llm_decide", 0),
                        tokens_used=d_tokens,
                        input_summary="LLM 分析问题并决定调用工具",
                        output_summary=f"决定调用 {len(tools_called)} 个工具",
                        metadata={"decision": "call_tools", "tool_count": len(tools_called),
                                  "provider": d_provider})
            record_span(session_id, "tools", timings.get("tools", 0), tokens_used=0,
                        input_summary=f"执行 {len(tools_called)} 个 Skill",
                        output_summary="; ".join(t[:60] for t in tools_called[:3]),
                        metadata={"tool_calls": list(tool_names[:10]), "count": len(tools_called)})
        else:
            record_span(session_id, "llm_decide", timings.get("llm_decide", 0),
                        tokens_used=d_tokens,
                        input_summary="LLM 分析问题",
                        output_summary="直接回答（无需工具）",
                        metadata={"decision": "answer_directly", "provider": d_provider})

        # reflect
        iterations = result.get("iteration", 0)
        if iterations > 0:
            r_tokens, _ = _role_tokens("reflect")
            record_span(session_id, "reflect", timings.get("reflect", 0),
                        tokens_used=r_tokens,
                        input_summary=f"校验工具返回数据 — 第{iterations}轮",
                        output_summary="数据完整，进入回答" if iterations == 1 else f"经过{iterations}轮迭代",
                        metadata={"iterations": iterations})

        reply = extract_ai_response(result["messages"])
        products = extract_products(result.get("skill_results", []))
        charts = build_charts_from_skills(result.get("skill_results", []), _seg_names_safe())

        # ── Observability recording ──
        try:
            from agent.observability import record_request
            tools_called_names = [
                getattr(sr, 'summary', '')[:60] or str(sr)[:60]
                for sr in skill_results
            ]
            fcr = result.get("fact_check_result")
            fc_passed = fcr.passed if fcr else None
            total_tokens = sum(
                sum(u.total_tokens for u in usages)
                for usages in usage_bucket.values()
            )
            record_request(
                session_id=session_id,
                question=question[:100],
                reply_len=len(reply),
                duration_ms=total_elapsed,
                tokens=total_tokens,   # real token count from adapter usage
                tools_called=tools_called_names,
                fact_check_passed=fc_passed,
            )
        except Exception:
            pass  # observability is best-effort

        # respond
        record_span(session_id, "respond", timings.get("respond", 0), tokens_used=0,
                    input_summary="基于工具数据生成最终回答",
                    output_summary=reply[:200])

        # fact_check
        fcr = result.get("fact_check_result")
        fc_passed = None
        v_count = 0
        if fcr:
            fc_passed = fcr.passed
            v_count = len(fcr.violations) if hasattr(fcr, 'violations') else 0
            record_span(session_id, "fact_check", timings.get("fact_check", 0), tokens_used=0,
                        input_summary=f"三层规则校验 ({len(reply)}字回复)",
                        output_summary="✅ 通过" if fc_passed else f"⚠️ {v_count}个违规",
                        metadata={"passed": fc_passed, "violations": v_count})

        biz_mem = get_business_memory(session_id)
        biz_mem.analysis_state = "completed"

        finish_trace(session_id, reply, fact_check_passed=fc_passed, violation_count=v_count)

        elapsed = round((time.monotonic() - start) * 1000, 2)
        logger.info("agent_response", extra={
            "session_id": session_id,
            "question_len": len(question),
            "reply_len": len(reply),
            "duration_ms": elapsed,
            "iterations": iterations,
            "check_mode": mode,
            "tools_called": len(tools_called),
        })
        return reply, iterations, products, charts

    except APIError:
        finish_trace(session_id, "API Error", fact_check_passed=False)
        raise
    except Exception as e:
        err_elapsed = round((time.monotonic() - start) * 1000, 2)
        logger.error("agent_error", extra={
            "session_id": session_id,
            "error": str(e),
        })
        record_span(session_id, "respond", err_elapsed, tokens_used=0,
                    input_summary=question[:100],
                    output_summary=f"❌ 执行失败: {str(e)[:200]}")
        finish_trace(session_id, f"Error: {e}", fact_check_passed=False)
        raise APIError(
            f"Agent invocation failed: {e}",
            context={"session_id": session_id},
        ) from e


def ask_agent(
    question: str,
    session_id: str = "default",
    check_mode: str | None = None,
    max_tool_rounds: int | None = None,
    system_override: str | None = None,
) -> str:
    """Invoke the Agentic RAG graph with session persistence.

    Args:
        question: User's question in natural language (Chinese).
        session_id: Conversation session identifier (maps to thread_id).
        check_mode: ``\"relaxed\"`` (user chat, annotate warnings) or
                    ``\"strict\"`` (autonomous analysis, block + retry).
                    Defaults to ``FACT_CHECK_MODE`` from settings.
        max_tool_rounds: Max tool-call iterations. Defaults to
                    ``MAX_REFLECTION_ROUNDS``.
        system_override: Optional role prompt replacing the scenario
                    template (multi-agent orchestrator).

    Returns:
        Agent's reply text.

    Raises:
        APIError: if the agent invocation fails or produces no AI response.
    """
    reply, _, _, _ = _ask_agent_internal(
        question, session_id, check_mode, max_tool_rounds, system_override,
    )
    return reply


async def _ask_agent_stream(
    question: str,
    session_id: str = "default",
    check_mode: str | None = None,
    max_tool_rounds: int | None = None,
):
    """Async generator: real-time SSE events for the whole agent execution.

    Consumes the graph via astream with three stream modes:
      - "updates" → node results (node_end details, fact_check, final answer)
      - "messages" → token-level deltas of the llm_decide answer stream
      - "custom"   → node_start / tool_call / tool_result events emitted by nodes

    Yields dict events (not SSE frames — api/routes.py formats them):
      meta / node_start / node_end_detail / tool_call / tool_result /
      delta / fact_check / answer / done / error
    """
    import asyncio

    settings = get_settings()
    mode = check_mode or settings.FACT_CHECK_MODE
    max_rounds = max_tool_rounds or settings.MAX_REFLECTION_ROUNDS

    yield {"event": "meta", "session_id": session_id, "check_mode": mode}

    # 流式路径同样记录 Trace(实测耗时 + 适配器真实 token)
    from agent.tracer import start_trace, record_span, finish_trace
    trace = start_trace(session_id, question)
    usage_bucket = start_usage_collection()

    input_state = {
        "messages": [{"role": "user", "content": question}],
        "session_id": session_id,
        "check_mode": mode,
        "max_tool_rounds": max_rounds,
        "force_analysis": False,
    }

    t_start = time.monotonic()
    final_reply = ""
    iterations = 0
    tools_called: list[str] = []
    timings: dict[str, float] = {}
    fc_passed: bool | None = None
    fc_violations = 0
    last_skill_results: list = []

    def _role_tokens(role: str) -> int:
        return sum(u.total_tokens for u in usage_bucket.get(role, []))

    try:
        async for stream_mode, payload in _agent_graph.astream(
            input_state,
            config={"configurable": {"thread_id": session_id}},
            stream_mode=["updates", "messages", "custom"],
        ):
            if stream_mode == "messages":
                chunk, meta = payload
                node = (meta or {}).get("langgraph_node", "")
                if node == "llm_decide" and isinstance(chunk, AIMessageChunk):
                    if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                        yield {"event": "delta", "node": node, "content": chunk.content}
            elif stream_mode == "custom":
                yield payload
                if payload.get("event") == "tool_call":
                    tools_called.append(payload.get("name", ""))
            elif stream_mode == "updates":
                for node, update in (payload or {}).items():
                    for n, ms in (update.get("node_timings") or {}).items():
                        timings[n] = ms
                    if "skill_results" in update:
                        last_skill_results = update["skill_results"]
                    if "messages" in update:
                        text = extract_last_ai_text(update)
                        if text:
                            final_reply = text
                    if node == "fact_check":
                        fcr = update.get("fact_check_result")
                        # fact_check 节点返回独立的 fact_check_retries 计数
                        # (reflect 不执行时 iteration 不会递增,读它会恒为 0)
                        iterations = update.get("fact_check_retries", 0)
                        fc_passed = bool(fcr and fcr.passed)
                        fc_violations = len(getattr(fcr, "violations", []) or [])
                        yield {
                            "event": "fact_check",
                            "passed": fc_passed,
                            "violations": fc_violations,
                            "mode": mode,
                        }

        answer = final_reply or "[系统] 未能生成回答，请重试。"
        products = extract_products(last_skill_results)
        charts = build_charts_from_skills(last_skill_results, _seg_names_safe())

        # ── Record trace spans (real measured durations + adapter tokens) ──
        span_nodes = ["preprocess", "llm_decide", "tools", "reflect", "respond", "fact_check"]
        span_tokens = {"preprocess": "preprocess", "llm_decide": "decide",
                       "tools": "", "reflect": "reflect", "respond": "", "fact_check": ""}
        for node in span_nodes:
            ms = timings.get(node, 0)
            role = span_tokens[node]
            tokens_used = _role_tokens(role) if role else 0
            if node not in timings and not tokens_used:
                continue   # 节点未执行且无 token → 不落 span

            record_span(session_id, node, ms, tokens_used=tokens_used,
                        input_summary=question[:100],
                        output_summary=(answer[:200] if node in ("respond", "fact_check") else ""),
                        metadata={"stream": True})
        finish_trace(session_id, answer,
                     fact_check_passed=fc_passed, violation_count=fc_violations)

        yield {"event": "answer", "content": answer}
        yield {
            "event": "done",
            "reply": answer,
            "elapsed_ms": round((time.monotonic() - t_start) * 1000, 1),
            "iterations": iterations,
            "tools_called": tools_called,
            "products": products,
            "charts": charts,
        }
    except asyncio.CancelledError:
        raise   # 客户端断开:图在 executor 中会跑完并写 checkpoint,状态一致
    except Exception as e:
        logger.error("stream_error", extra={
            "session_id": session_id, "error": str(e),
        })
        finish_trace(session_id, f"Error: {e}", fact_check_passed=False)
        yield {"event": "error", "message": "服务器内部错误"}
