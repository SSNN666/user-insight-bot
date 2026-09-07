"""统一适配器 ↔ LangGraph 桥接层。

把 llm/adapter.py 的 FallbackChain 包装成 LangChain BaseChatModel,
使 Agent 图获得:超时/429/额度/上下文超长处理、多供应商自动降级、
token 统计,同时保留 LangChain 原生 bind_tools 语义。

设计要点:
- bind_tools 必须自实现(langchain-core 0.2.43 基类直接 raise NotImplementedError),
  用 convert_to_openai_tool 把 StructuredTool 转 OpenAI schema 存实例属性,
  copy() 返回新实例(避免 RunnableBinding 包装的兼容坑)。
- 消息经 convert_to_openai_messages 转 OpenAI 协议(实测 tool_calls/ToolMessage 转换正确)。
- usage 写入标准 usage_metadata + response_metadata["token_usage"],tracer 可直接读。
- per-run usage 收集用 ContextVar:preprocess/reflect 的调用不进消息流,
  通过 ContextVar 汇总到 _ask_agent_internal 的 trace 统计。
- FallbackChain 构造后只读共享(线程安全);FallbackChatModel 每次轻量新建。
"""

import threading
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from config.settings import get_settings
from llm.adapter import FallbackChain, StreamChunk, UsageInfo, build_llm
from log.logger import get_logger

logger = get_logger(__name__)

# per-run usage 收集:角色 → [UsageInfo]。_ask_agent_internal 入口 set,节点线程经
# langgraph 的 copy_context 继承(ContextVar 跨线程安全)。
_usage_ctx: ContextVar[Optional[dict[str, list[UsageInfo]]]] = ContextVar(
    "llm_bridge_usage", default=None
)


def start_usage_collection() -> dict[str, list[UsageInfo]]:
    """开启一轮 usage 收集,返回收集容器(调用方负责读取)。"""
    bucket: dict[str, list[UsageInfo]] = {}
    _usage_ctx.set(bucket)
    return bucket


def _record_usage(role: str, usage: UsageInfo) -> None:
    bucket = _usage_ctx.get()
    if bucket is not None:
        bucket.setdefault(role, []).append(usage)


def _log_usage(role: str, usage: UsageInfo) -> None:
    _record_usage(role, usage)
    logger.info("llm_usage", extra={
        "role": role,
        "provider": usage.provider,
        "model": usage.model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "latency_ms": usage.latency_ms,
    })


class FallbackChatModel(BaseChatModel):
    """包装 FallbackChain 的 LangChain chat model。

    绑定工具时返回新实例(共享同一个无状态 chain),保持 isinstance 语义为
    chat model,兼容 langgraph 的 tools_condition 与 messages 模式。
    """

    chain: Any = None               # FallbackChain(构造后只读共享)
    tools: Optional[list[dict]] = None      # OpenAI 工具 schema(bind_tools 填充)
    tool_choice: Optional[str] = None
    model_name: str = ""

    @property
    def _llm_type(self) -> str:
        return "fallback-chain"

    def bind_tools(
        self, tools: list, *, tool_choice: Optional[str] = None, **kwargs: Any
    ) -> "FallbackChatModel":
        """StructuredTool → OpenAI 工具 schema,存实例属性而非 RunnableBinding。"""
        from langchain_core.utils.function_calling import convert_to_openai_tool
        schemas = [convert_to_openai_tool(t) for t in tools]
        return self.model_copy(update={"tools": schemas, "tool_choice": tool_choice})

    # ── 非流式 ──────────────────────────────────────────────

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from langchain_core.messages import convert_to_openai_messages
        payload = convert_to_openai_messages(messages)
        resp = self.chain.invoke(
            payload, tools=self.tools, tool_choice=self.tool_choice, stop=stop,
        )
        usage = resp.usage
        ai = AIMessage(
            content=resp.content or "",
            tool_calls=resp.tool_calls or [],
            response_metadata=self._response_metadata(resp),
            usage_metadata=self._usage_metadata(usage),
        )
        return ChatResult(generations=[ChatGeneration(message=ai)], llm_output={})

    # ── 流式 ────────────────────────────────────────────────

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        from langchain_core.messages import convert_to_openai_messages
        payload = convert_to_openai_messages(messages)
        emitted = False   # 任何输出(文本或工具调用)都算"已发出"
        for ev in self.chain.stream_events(
            payload, tools=self.tools, tool_choice=self.tool_choice, stop=stop,
        ):
            if ev.text:
                emitted = True
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content=ev.text,
                    response_metadata=(
                        {"fallback_switch": True} if ev.fallback_switch else {}
                    ),
                ))
            if ev.tool_calls or ev.finish_reason or ev.usage:
                emitted = True
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="",
                    tool_calls=ev.tool_calls or [],
                    usage_metadata=self._usage_metadata(ev.usage),
                    response_metadata=self._finish_metadata(ev),
                ))
        if not emitted:
            # 兜底:所有供应商流式首块前全挂 → 走一次非流式生成
            gen = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            message = gen.generations[0].message
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=message.content,
                tool_calls=getattr(message, "tool_calls", None) or [],
                usage_metadata=getattr(message, "usage_metadata", None),
                response_metadata=getattr(message, "response_metadata", None) or {},
            ))

    # ── metadata 构造 ───────────────────────────────────────

    @staticmethod
    def _usage_metadata(usage: Optional[UsageInfo]) -> Optional[dict]:
        if usage is None:
            return None
        return {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    @staticmethod
    def _response_metadata(resp) -> dict:
        meta: dict = {"provider": None, "fallback": resp.fallback}
        if resp.usage:
            meta["model_name"] = resp.usage.model
            meta["provider"] = resp.usage.provider
            meta["token_usage"] = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
                "latency_ms": resp.usage.latency_ms,
            }
        if resp.error_kind:
            meta["error_kind"] = resp.error_kind
        return meta

    @staticmethod
    def _finish_metadata(ev: StreamChunk) -> dict:
        meta: dict = {"provider": ev.provider}
        if ev.finish_reason:
            meta["finish_reason"] = ev.finish_reason
        if ev.usage:
            meta["model_name"] = ev.usage.model
            meta["token_usage"] = {
                "prompt_tokens": ev.usage.prompt_tokens,
                "completion_tokens": ev.usage.completion_tokens,
                "total_tokens": ev.usage.total_tokens,
                "latency_ms": ev.usage.latency_ms,
            }
        return meta


# ── 工厂 ───────────────────────────────────────────────────────

_chains: dict[str, FallbackChain] = {}
_chains_lock = threading.Lock()


def _get_chain(role: str) -> FallbackChain:
    """惰性构建 + 缓存角色降级链(构造后只读共享,线程安全)。"""
    chain = _chains.get(role)
    if chain is not None:
        return chain
    with _chains_lock:
        chain = _chains.get(role)
        if chain is None:
            settings = get_settings()
            chain = build_llm(
                role=role,
                settings=settings,
                on_usage=_log_usage,
                temperature=settings.LLM_TEMPERATURE,   # 沿用全局温度配置
            )
            _chains[role] = chain
            logger.info("fallback_chain_built", extra={
                "role": role, "providers": chain.active_providers,
            })
    return chain


def get_fallback_llm(
    role: str = "decide", *, tool_names: Optional[list[str]] = None,
) -> FallbackChatModel:
    """构建角色 LLM。tool_names 非 None 时按名单裁剪工具(物理隔离)。"""
    chain = _get_chain(role)
    llm = FallbackChatModel(chain=chain)
    if tool_names is not None:
        # tools 模块导入副作用:确保 Skill 一次性注册(向后兼容);
        # 工具列表实时从注册表拉取(热加载新增的 Skill 即时进入 LLM 工具集,
        # 不再依赖 import 时的静态快照)
        from tools import tools as _registration_trigger  # noqa: F401
        from skills import SkillRegistry
        scoped = [t for t in SkillRegistry.get_langchain_tools()
                  if t.name in tool_names]
        return llm.bind_tools(scoped)
    return llm


def reset_fallback_chains() -> None:
    """清空链缓存(测试/配置热更)。"""
    with _chains_lock:
        _chains.clear()
