"""llm/bridge 单测:FallbackChatModel 的 bind_tools / 消息生成 / usage 回流。

使用 mock 适配器注入 FallbackChain,不发任何真实 LLM 请求。
"""
import pytest

from llm.adapter import BaseLLMAdapter, FallbackChain, LLMResponse, UsageInfo
from llm.bridge import FallbackChatModel, start_usage_collection, _record_usage


class _MockAdapter(BaseLLMAdapter):
    name = "mock"

    def __init__(self, content="ok", tool_calls=None):
        self._content = content
        self._tool_calls = tool_calls
        self.invoke_calls = []

    def invoke(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None,
               tools=None, tool_choice=None, stop=None):
        self.invoke_calls.append({
            "messages": messages, "tools": tools, "stop": stop,
        })
        return LLMResponse(
            content=self._content,
            usage=UsageInfo(prompt_tokens=11, completion_tokens=7,
                            total_tokens=18, model="mock-model", provider="mock"),
            tool_calls=self._tool_calls,
        )

    def stream(self, messages, *, temperature=0.7, max_tokens=None,
               timeout=None, request_id=None):
        yield self._content


def _chain(**kw) -> FallbackChain:
    return FallbackChain([_MockAdapter(**kw)], max_retries=0)


class TestBindTools:
    def test_bind_tools_schemas(self):
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class _Args(BaseModel):
            keyword: str = Field(default="", description="关键词")

        def _fn(keyword: str = "") -> str:
            return keyword

        tool = StructuredTool.from_function(
            func=_fn, name="search_products", description="搜索商品",
            args_schema=_Args,
        )
        llm = FallbackChatModel(chain=_chain())
        bound = llm.bind_tools([tool])
        assert bound is not llm                      # copy 语义
        assert len(bound.tools) == 1
        schema = bound.tools[0]["function"]
        assert schema["name"] == "search_products"
        assert "keyword" in schema["parameters"]["properties"]


class TestGenerate:
    def test_generate_passes_tools_and_parses_usage(self):
        llm = FallbackChatModel(chain=_chain(content="表格式回答")).bind_tools([])
        result = llm.invoke([{"role": "user", "content": "各分群人数?"}])
        assert result.content == "表格式回答"
        assert result.usage_metadata["total_tokens"] == 18
        assert result.response_metadata["provider"] == "mock"
        assert result.response_metadata["token_usage"]["total_tokens"] == 18

    def test_generate_tool_calls(self):
        chain = _chain(
            content="",
            tool_calls=[{"name": "get_user_segment_stats", "args": {}, "id": "call-1"}],
        )
        llm = FallbackChatModel(chain=chain)
        result = llm.invoke([{"role": "user", "content": "各分群人数?"}])
        assert result.tool_calls[0]["name"] == "get_user_segment_stats"
        assert result.tool_calls[0]["id"] == "call-1"

    def test_stream_yields_text_and_usage(self):
        llm = FallbackChatModel(chain=_chain(content="你好"))
        chunks = list(llm.stream([{"role": "user", "content": "hi"}]))
        text = "".join(c.content for c in chunks if c.content)
        assert text == "你好"


class TestUsageContext:
    def test_contextvar_collection(self):
        bucket = start_usage_collection()
        _record_usage("decide", UsageInfo(total_tokens=100, provider="mock"))
        assert bucket["decide"][0].total_tokens == 100


class TestFactory:
    def test_get_fallback_llm_scopes_tools(self, monkeypatch):
        from llm.bridge import get_fallback_llm, reset_fallback_chains
        from config.settings import get_settings

        # 隔离测试配置:云 Key 置空 + 禁用本地兜底 → 空链(不发真实请求)
        monkeypatch.setattr(get_settings(), "DASHSCOPE_API_KEY", "")
        monkeypatch.setattr(get_settings(), "DEEPSEEK_API_KEY", "")
        monkeypatch.setattr(get_settings(), "QIANFAN_API_KEY", "")
        monkeypatch.setattr(get_settings(), "LLM_LOCAL_FALLBACK_ENABLED", False)
        reset_fallback_chains()

        llm = get_fallback_llm("decide", tool_names=["search_products"])
        assert isinstance(llm, FallbackChatModel)
        names = [t["function"]["name"] for t in llm.tools]
        assert names == ["search_products"]
        reset_fallback_chains()
