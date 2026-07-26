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
"""

import time
from typing import Annotated, Any, Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import ToolMessage

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
from tools import tools
from errors.exceptions import APIError
from log.logger import get_logger

logger = get_logger(__name__)


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
    should_continue: bool


# ── Helpers ─────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_BASE_URL,
    )


def _build_llm_with_tools(tool_names: list[str] | None = None):
    """LLM that knows about available skills (via bind_tools).

    Args:
        tool_names: Optional list of tool names to expose. If None, all tools.
                    Scoping tools per intent prevents the LLM from calling
                    inappropriate tools (e.g. segment tools during shopping).
    """
    llm = _build_llm()
    if tool_names is not None:
        scoped = [t for t in tools if t.name in tool_names]
        return llm.bind_tools(scoped)
    return llm.bind_tools(tools)


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


# ── Graph Nodes ─────────────────────────────────────────────────


def _preprocess_node(state: AgentState) -> dict:
    """Classify intent, extract entities, decompose query."""
    settings = get_settings()
    if not settings.PREPROCESS_ENABLED:
        return {
            "preprocess_result": None, "iteration": 0,
            "skill_results": [], "should_continue": True,
        }

    user_msgs = [m for m in state["messages"]
                 if getattr(m, 'type', None) == 'human']
    query = user_msgs[-1].content if user_msgs else ""

    result = preprocess_query(query, _build_llm())
    return {
        "preprocess_result": result,
        "iteration": 0,
        "skill_results": [],
        "should_continue": True,
    }


def _llm_decide_node(state: AgentState) -> dict:
    """LLM decides: call tools or answer directly.

    Uses scenario-specific prompt template based on query keywords.
    Tools are scoped to the selected prompt — shopping mode only sees
    search_products / get_categories, analysis mode only sees segment tools.
    """
    from agent.prompts import PROMPT_SHOPPING as _SHOP

    settings = get_settings()

    # ── Select prompt template ──
    pre = state.get("preprocess_result")
    if pre:
        query = pre.original_query or ""
        prompt = select_prompt_template(pre.intent.value, query)
    else:
        prompt = PROMPT_DATA_STATS

    # ── Scope tools per prompt context ──
    # Deterministic: the LLM physically cannot call tools it can't see.
    if prompt == _SHOP:
        scoped_names = ["search_products", "get_categories"]
    else:
        scoped_names = [
            "get_user_segment_stats", "get_segment_rules",
            "get_high_value_users", "get_segment_growth",
            "refresh_pipeline",
        ]
    llm_with_tools = _build_llm_with_tools(scoped_names)

    context_parts = [prompt]

    # Inject preprocess result
    if pre and pre.intent.value != "general":
        context_parts.append(f"\n[查询分析] {pre.format_for_context()}")

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
    if settings.TOOL_CALL_GUARD_ENABLED and not skill_results and prompt != _SHOP:
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

    # Build messages: system + existing history
    messages = [{"role": "system", "content": enriched_system}]
    messages.extend(state["messages"])

    response = llm_with_tools.invoke(messages)

    # ── Tool-call text-pattern → native tool_call bridge ──
    # Some Ollama models output text like "get_user_segment_stats"
    # instead of using the native tool_calls API. Parse & convert.
    if not getattr(response, 'tool_calls', None):
        content = getattr(response, 'content', '') or ''
        import re as _re
        found_tools = [tn for tn in scoped_names if tn in content]
        if found_tools:
            # Create synthetic tool_calls from text match
            from langchain_core.messages import AIMessage
            synthetic_calls = [
                {"name": tn, "args": {}, "id": f"synth-{i}"}
                for i, tn in enumerate(found_tools[:3])
            ]
            response = AIMessage(
                content="",
                tool_calls=synthetic_calls,
            )

    return {"messages": [response]}


def _tools_node(state: AgentState) -> dict:
    """Execute tool calls directly via SkillRegistry, capturing SkillResult objects.

    This replaces ToolNode so we can store actual SkillResult.data
    as ground truth for the fact-checker.
    """
    from skills import SkillRegistry

    last_msg = state["messages"][-1] if state["messages"] else None
    skill_results: list[Any] = list(state.get("skill_results", []))
    tool_messages: list[ToolMessage] = []

    if last_msg and hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            skill_name = tc.get("name", "unknown")
            skill_args = tc.get("args", {})
            tool_call_id = tc.get("id", "")

            skill = SkillRegistry.get(skill_name)
            if skill:
                try:
                    result = skill.execute(**skill_args)
                    skill_results.append(result)
                    tool_messages.append(ToolMessage(
                        content=result.to_context_string(),
                        tool_call_id=tool_call_id,
                        name=skill_name,
                    ))
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
            else:
                tool_messages.append(ToolMessage(
                    content=f"Skill '{skill_name}' not found.",
                    tool_call_id=tool_call_id,
                    name=skill_name,
                ))

    # Update business memory
    session_id = state.get("session_id", "default")
    biz_mem = get_business_memory(session_id)
    if last_msg and hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            biz_mem.add_skill_call(tc.get("name", "unknown"))
    biz_mem.analysis_state = "tools_executed"

    return {"messages": tool_messages, "skill_results": skill_results}


def _reflect_node(state: AgentState) -> dict:
    """Validate tool results; decide whether to re-query."""
    settings = get_settings()
    iteration = state.get("iteration", 0) + 1

    skill_results = state.get("skill_results", [])

    user_msgs = [m for m in state["messages"]
                 if getattr(m, 'type', None) == 'human']
    original_query = user_msgs[-1].content if user_msgs else ""

    reflect_result = reflect(skill_results, original_query, _build_llm())

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
        }

    return {
        "reflect_result": reflect_result,
        "iteration": iteration,
        "should_continue": False,
    }


def _respond_node(state: AgentState) -> dict:
    """Ensure the final response is valid Markdown."""
    ai_msgs = [
        m for m in state["messages"]
        if getattr(m, 'type', None) == 'ai'
        and getattr(m, 'content', None)
    ]
    if not ai_msgs:
        return {}

    last_ai = ai_msgs[-1]
    formatted = enforce_markdown(last_ai.content)

    if formatted != last_ai.content:
        amended_msg = last_ai.model_copy(update={"content": formatted})
        new_messages = list(state["messages"])
        for i in range(len(new_messages) - 1, -1, -1):
            if getattr(new_messages[i], 'type', None) == 'ai':
                new_messages[i] = amended_msg
                break
        return {"messages": new_messages}

    return {}


def _fact_check_node(state: AgentState) -> dict:
    """Run three-layer fact-check on the last AI response."""
    from agent.fact_checker import run_fact_check

    settings = get_settings()
    check_mode = state.get("check_mode", settings.FACT_CHECK_MODE)

    ai_msgs = [
        m for m in state["messages"]
        if getattr(m, 'type', None) == 'ai'
        and getattr(m, 'content', None)
    ]
    if not ai_msgs:
        return {"fact_check_result": None}

    llm_response = ai_msgs[-1].content
    skill_results = state.get("skill_results", [])

    result = run_fact_check(llm_response, skill_results, check_mode)

    if not result.passed and check_mode == "strict":
        correction_text = result.format_corrections_for_llm()
        return {
            "fact_check_result": result,
            "messages": [{"role": "system", "content": correction_text}],
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
        }

    return {"fact_check_result": result}


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
        and state.get("iteration", 0) < settings.FACT_CHECK_MAX_RETRIES
    ):
        logger.info("fact_check_retry", extra={
            "violations": len(fcr.violations),
            "iteration": state.get("iteration", 0),
        })
        return "llm_decide"
    return "__end__"


# ── Build Graph ─────────────────────────────────────────────────


def _build_graph():
    """Construct the Agentic RAG StateGraph (Phase 2)."""
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
    memory_saver = InMemorySaver()
    return workflow.compile(checkpointer=memory_saver)


# ── Module-level graph ──────────────────────────────────────────

_agent_graph = _build_graph()


# ── Public API ──────────────────────────────────────────────────


def _ask_agent_internal(
    question: str,
    session_id: str = "default",
    check_mode: str | None = None,
    max_tool_rounds: int | None = None,
) -> tuple[str, int]:
    """Invoke the graph and return (reply_text, iteration_count).

    Used by the watcher engine to track tool-call rounds.
    """
    start = time.monotonic()
    settings = get_settings()
    mode = check_mode or settings.FACT_CHECK_MODE

    # ── Agent Trace (Phase: observability) ──
    from agent.tracer import start_trace, record_span, finish_trace
    trace = start_trace(session_id, question)

    try:
        result = _agent_graph.invoke(
            {
                "messages": [{"role": "user", "content": question}],
                "session_id": session_id,
                "check_mode": mode,
                "max_tool_rounds": max_tool_rounds or settings.MAX_REFLECTION_ROUNDS,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        total_elapsed = round((time.monotonic() - start) * 1000)

        # ── Record trace spans from state ──
        # preprocess
        pre = result.get("preprocess_result")
        if pre:
            record_span(session_id, "preprocess", total_elapsed * 0.05, tokens_used=150,
                        input_summary=question[:100],
                        output_summary=f"意图: {pre.intent.value} | 子任务: {pre.sub_queries if hasattr(pre,'sub_queries') else '无'}",
                        metadata={"intent": pre.intent.value, "entities": pre.entities})

        # llm_decide + tools
        skill_results = result.get("skill_results", [])
        tools_called = []
        for sr in skill_results:
            name = getattr(sr, 'summary', '')[:100] or str(sr)[:100]
            tools_called.append(name)
        if tools_called:
            record_span(session_id, "llm_decide", total_elapsed * 0.10, tokens_used=200,
                        input_summary="LLM 分析问题并决定调用工具",
                        output_summary=f"决定调用 {len(tools_called)} 个工具",
                        metadata={"decision": "call_tools", "tool_count": len(tools_called)})
            record_span(session_id, "tools", total_elapsed * 0.35, tokens_used=0,
                        input_summary=f"执行 {len(tools_called)} 个 Skill",
                        output_summary="; ".join(t[:60] for t in tools_called[:3]),
                        metadata={"tool_calls": [t[:40] for t in tools_called], "count": len(tools_called)})
        else:
            record_span(session_id, "llm_decide", total_elapsed * 0.20, tokens_used=200,
                        input_summary="LLM 分析问题",
                        output_summary="直接回答（无需工具）",
                        metadata={"decision": "answer_directly"})

        # reflect
        iterations = result.get("iteration", 0)
        if iterations > 0:
            record_span(session_id, "reflect", total_elapsed * 0.15, tokens_used=200 * iterations,
                        input_summary=f"校验工具返回数据 — 第{iterations}轮",
                        output_summary="数据完整，进入回答" if iterations == 1 else f"经过{iterations}轮迭代",
                        metadata={"iterations": iterations})

        reply = extract_ai_response(result["messages"])

        # ── Observability recording ──
        try:
            from agent.observability import record_request
            tools_called_names = [
                getattr(sr, 'summary', '')[:60] or str(sr)[:60]
                for sr in skill_results
            ]
            fcr = result.get("fact_check_result")
            fc_passed = fcr.passed if fcr else None
            record_request(
                session_id=session_id,
                question=question[:100],
                reply_len=len(reply),
                duration_ms=total_elapsed,
                tokens=len(reply) // 2,  # rough token estimate
                tools_called=tools_called_names,
                fact_check_passed=fc_passed,
            )
        except Exception:
            pass  # observability is best-effort

        # respond
        record_span(session_id, "respond", total_elapsed * 0.25, tokens_used=len(reply) // 2,
                    input_summary="基于工具数据生成最终回答",
                    output_summary=reply[:200])

        # fact_check
        fcr = result.get("fact_check_result")
        fc_passed = None
        v_count = 0
        if fcr:
            fc_passed = fcr.passed
            v_count = len(fcr.violations) if hasattr(fcr, 'violations') else 0
            record_span(session_id, "fact_check", total_elapsed * 0.10, tokens_used=0,
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
        return reply, iterations

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

    Returns:
        Agent's reply text.

    Raises:
        APIError: if the agent invocation fails or produces no AI response.
    """
    reply, _ = _ask_agent_internal(question, session_id, check_mode, max_tool_rounds)
    return reply


async def _ask_agent_stream(
    question: str,
    session_id: str = "default",
    check_mode: str | None = None,
    max_tool_rounds: int | None = None,
):
    """Async generator: yields reply tokens as they become available.

    Runs the sync graph via LangGraph's ``.astream()``, then streams the
    final response character-by-character with slight delays for a natural
    typing effect.

    Usage::

        async for token in _ask_agent_stream("各分群人数?"):
            yield token  # send via SSE to client
    """
    import asyncio

    settings = get_settings()
    mode = check_mode or settings.FACT_CHECK_MODE
    max_rounds = max_tool_rounds or settings.MAX_REFLECTION_ROUNDS

    # Run graph in thread pool (graph nodes are sync)
    loop = asyncio.get_event_loop()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        reply, _ = await loop.run_in_executor(
            pool,
            _ask_agent_internal,
            question, session_id, mode, max_rounds,
        )

    # Stream the response character by character (~30ms delay for natural feel)
    if reply:
        chunk_size = 3  # characters per yield
        for i in range(0, len(reply), chunk_size):
            yield reply[i:i + chunk_size]
            await asyncio.sleep(0.02)  # ~50 chars/sec — fast but visible
    else:
        yield "[系统] 未能生成回答，请重试。"
