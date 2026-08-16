"""Agent 图单元测试:剧本化适配器注入 FallbackChain,走真实桥接与图节点。

覆盖三条主路径:直接回答 / 工具调用 → 事实核查 / 反思回环补查。
不发任何真实 LLM 请求。
"""
import asyncio
import json

import pytest

from llm.adapter import BaseLLMAdapter, FallbackChain, LLMResponse, StreamChunk, UsageInfo
from llm.bridge import FallbackChatModel

from agent.agent import _agent_graph
from skills.base import SkillResult, SkillStatus


# ── 剧本化适配器 ──────────────────────────────────────────────

class _ScriptedAdapter(BaseLLMAdapter):
    """按脚本队列吐响应。

    注意:langchain-core 0.2.43 的 invoke 也经 _stream 合并(实测)→ 所有 LLM 调用
    都走 stream_events,故脚本条目统一支持 LLMResponse 与 dict 两种格式。
    """

    name = "scripted"

    def __init__(self, script: list):
        self._script = list(script)
        self.invoke_calls = 0
        self.stream_calls = 0

    def invoke(self, messages, **kw):
        self.invoke_calls += 1
        item = self._script.pop(0)
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(content=item.get("text", ""),
                           usage=item.get("usage") or _usage(),
                           tool_calls=item.get("tool_calls"))

    def stream_events(self, messages, **kw):
        self.stream_calls += 1
        item = self._script.pop(0)
        if isinstance(item, LLMResponse):
            text, tool_calls, usage = item.content or "", item.tool_calls, item.usage
        else:
            text, tool_calls = item.get("text", ""), item.get("tool_calls")
            usage = item.get("usage") or _usage()
        for i in range(0, len(text), 3):   # 3 字符一块,模拟 token 级增量
            yield StreamChunk(text=text[i:i + 3], provider="scripted")
        yield StreamChunk(
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=usage,
            provider="scripted",
        )

    def stream(self, messages, **kw):
        for ev in self.stream_events(messages, **kw):
            if ev.text:
                yield ev.text


def _usage() -> UsageInfo:
    return UsageInfo(prompt_tokens=5, completion_tokens=5, total_tokens=10,
                     model="scripted", provider="scripted")


def _llm_with_script(script: list, monkeypatch) -> None:
    """把剧本适配器包装成 FallbackChatModel,替换 agent 的 LLM 工厂。"""
    chain = FallbackChain([_ScriptedAdapter(script)], max_retries=0)
    model = FallbackChatModel(chain=chain)

    import agent.agent as agent_mod
    monkeypatch.setattr(agent_mod, "get_fallback_llm", lambda *a, **kw: model)
    return model


def _fake_skill(data, summary="分群统计"):
    class _Fake:
        def execute(self, **kwargs):
            return SkillResult(status=SkillStatus.SUCCESS, data=data,
                               summary=summary, confidence=0.95)
    return _Fake()


def _invoke_graph(question: str, check_mode: str = "relaxed") -> dict:
    return _agent_graph.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "session_id": f"graph-test-{abs(hash(question))}",
            "check_mode": check_mode,
            "max_tool_rounds": 2,
        },
        config={"configurable": {"thread_id": f"graph-test-{abs(hash(question))}"}},
    )


# ── 用例 ──────────────────────────────────────────────────────

SEG_STATS_DATA = [{"segment": 0, "用户数": 50, "平均近度": 10.0,
                   "平均频次": 2.0, "平均消费": 100.0}]


class TestDirectAnswer:
    def test_no_tools_direct_answer(self, monkeypatch):
        """无工具直接回答:preprocess(general)→ decide 纯文本 → respond → fact_check。"""
        _llm_with_script([
            LLMResponse(content='{"intent":"general","entities":{},"sub_queries":["你好"]}',
                        usage=_usage()),                                   # preprocess
            {"text": "你好！我是电商购物助手。", "tool_calls": None},     # decide(流式)
        ], monkeypatch)

        result = _invoke_graph("你好")
        reply = next(m for m in reversed(result["messages"])
                     if getattr(m, "type", None) == "ai" and m.content)
        assert "购物助手" in reply.content
        assert result["tools_called_names"] == []
        assert result["fact_check_result"].passed is True
        assert "llm_decide" in result["node_timings"]   # 实测耗时存在


class TestToolCallFlow:
    def test_tool_call_fact_check_pass(self, monkeypatch):
        """工具调用流:decide 调工具 → 假 Skill 返回数据 → 数据一致的答案过三层核查。"""
        import agent.agent as agent_mod
        monkeypatch.setattr(
            agent_mod, "SkillRegistry_get",
            lambda name: _fake_skill(SEG_STATS_DATA),
        )
        _llm_with_script([
            LLMResponse(content='{"intent":"user_segment","entities":{},"sub_queries":["各分群人数"]}',
                        usage=_usage()),                                   # preprocess
            {"text": "", "tool_calls": [
                {"name": "get_user_segment_stats", "args": {}, "id": "c1"},
            ]},                                                             # decide#1
            LLMResponse(content='{"is_complete":true,"has_conflict":false,"missing_info":[],'
                                '"conflict_details":"","suggestion":"","confidence":0.9}',
                        usage=_usage()),                                   # reflect
            {"text": "分群0有50人", "tool_calls": None},                   # decide#2(流式回答)
        ], monkeypatch)

        result = _invoke_graph("各分群的人数是多少？")
        assert result["tools_called_names"] == ["get_user_segment_stats"]
        assert len(result["skill_results"]) == 1
        assert result["iteration"] == 1
        fc = result["fact_check_result"]
        assert fc.passed is True
        reply = next(m for m in reversed(result["messages"])
                     if getattr(m, "type", None) == "ai" and m.content)
        assert "分群0有50人" in reply.content


class TestReflectLoop:
    def test_incomplete_data_triggers_requery(self, monkeypatch):
        """反思回环:数据不完整 → 建议补查 → 回到 llm_decide 再调工具 → 第二轮完成。"""
        import agent.agent as agent_mod

        calls = {"rules": 0, "stats": 0}

        class _FakeRulesSkill:
            def execute(self, **kwargs):
                calls["rules"] += 1
                return SkillResult(status=SkillStatus.SUCCESS,
                                   data="|--- recency <= 3.00\n|   |--- class: 0",
                                   summary="规则文本", confidence=0.9)

        class _FakeStatsSkill:
            def execute(self, **kwargs):
                calls["stats"] += 1
                return SkillResult(status=SkillStatus.SUCCESS, data=SEG_STATS_DATA,
                                   summary="分群统计", confidence=0.95)

        monkeypatch.setattr(agent_mod, "SkillRegistry_get",
                            lambda name: _FakeStatsSkill() if name == "get_user_segment_stats"
                            else _FakeRulesSkill())
        _llm_with_script([
            LLMResponse(content='{"intent":"user_segment","entities":{},"sub_queries":["各分群人数"]}',
                        usage=_usage()),                                   # preprocess
            {"text": "", "tool_calls": [
                {"name": "get_user_segment_stats", "args": {}, "id": "c1"},
            ]},                                                             # decide#1
            LLMResponse(content='{"is_complete":false,"has_conflict":false,'
                                '"missing_info":["分群规则"],"conflict_details":"",'
                                '"suggestion":"调用 get_segment_rules","confidence":0.4}',
                        usage=_usage()),                                   # reflect#1 不完整
            {"text": "", "tool_calls": [
                {"name": "get_segment_rules", "args": {}, "id": "c2"},
            ]},                                                             # decide#2 补查
            LLMResponse(content='{"is_complete":true,"has_conflict":false,"missing_info":[],'
                                '"conflict_details":"","suggestion":"","confidence":0.9}',
                        usage=_usage()),                                   # reflect#2 完整
            {"text": "分群0有50人", "tool_calls": None},                   # decide#3 回答
        ], monkeypatch)

        result = _invoke_graph("各分群的人数是多少？")
        assert calls["stats"] == 1 and calls["rules"] == 1     # 两轮各调一个工具
        assert result["tools_called_names"] == ["get_user_segment_stats", "get_segment_rules"]
        assert result["iteration"] == 2                        # 回环一轮


class TestExtractProducts:
    def test_product_skill_result_extracted(self):
        from agent.agent import extract_products
        sr = SkillResult(
            status=SkillStatus.SUCCESS,
            data=[{"product_id": 1, "product_name": "耳机", "category": "电子产品", "price": 299}],
            summary="商品列表", confidence=0.95,
        )
        assert extract_products([sr]) == [
            {"product_id": 1, "product_name": "耳机", "category": "电子产品", "price": 299},
        ]

    def test_non_product_results_ignored(self):
        from agent.agent import extract_products
        sr = SkillResult(status=SkillStatus.SUCCESS, data=SEG_STATS_DATA, summary="分群统计")
        assert extract_products([sr]) == []


class TestConversationSummary:
    def test_long_history_gets_summarized(self, monkeypatch):
        """消息数超阈值 → preprocess 生成摘要并写入 state,decide 仅见最近原文。"""
        from langchain_core.messages import HumanMessage, AIMessage

        _llm_with_script([
            LLMResponse(content='{"intent":"general","entities":{},"sub_queries":["继续"]}',
                        usage=_usage()),                                   # preprocess
            LLMResponse(content="历史摘要:用户反复询问分群数据。", usage=_usage()),  # 摘要调用
            {"text": "好的,继续为您服务。", "tool_calls": None},           # decide
        ], monkeypatch)

        history = []
        for i in range(11):
            history.append(HumanMessage(content=f"问题{i}"))
            history.append(AIMessage(content=f"回答{i}"))
        history.append(HumanMessage(content="继续"))

        result = _agent_graph.invoke(
            {"messages": history, "session_id": "summary-1",
             "check_mode": "relaxed", "max_tool_rounds": 2},
            config={"configurable": {"thread_id": "summary-1"}},
        )
        assert result["conversation_summary"] == "历史摘要:用户反复询问分群数据。"
        reply = next(m for m in reversed(result["messages"])
                     if getattr(m, "type", None) == "ai" and m.content)
        assert "继续为您服务" in reply.content


class TestForceAnalysis:
    def test_non_analysis_intent_falls_back_to_stats_prompt(self, monkeypatch):
        """管理台强制分析:非 user_segment 意图(购物路由)回退到数据统计人设。"""
        captured = {}

        class _CapturingAdapter(_ScriptedAdapter):
            def stream_events(self, messages, **kw):
                captured["decide_messages"] = messages
                return super().stream_events(messages, **kw)

        import agent.agent as agent_mod
        chain = FallbackChain([_CapturingAdapter([
            LLMResponse(content='{"intent":"general","entities":{},"sub_queries":[]}',
                        usage=_usage()),                                  # preprocess
            {"text": "运营分析回答", "tool_calls": None},                # decide
        ])], max_retries=0)
        monkeypatch.setattr(agent_mod, "get_fallback_llm",
                            lambda *a, **kw: FallbackChatModel(chain=chain))

        result = _agent_graph.invoke(
            {"messages": [{"role": "user", "content": "综合分析"}],
             "session_id": "force-1", "check_mode": "relaxed",
             "max_tool_rounds": 2, "force_analysis": True},
            config={"configurable": {"thread_id": "force-1"}},
        )
        content = "\n".join(
            str(getattr(m, "content", m)) for m in captured.get("decide_messages", [])
        )
        assert "电商数据分析师" in content        # 数据统计人设
        assert "你是电商购物助手" not in content  # 不再走购物路由(覆盖说明中的'购物助手'字样除外)
        assert "角色覆盖" in content              # 显式要求忽略历史购物人设

    def test_without_flag_keeps_shopping_prompt(self, monkeypatch):
        """对照组:无 force_analysis 时,general 意图保持购物助手人设。"""
        captured = {}

        class _CapturingAdapter(_ScriptedAdapter):
            def stream_events(self, messages, **kw):
                captured["decide_messages"] = messages
                return super().stream_events(messages, **kw)

        import agent.agent as agent_mod
        chain = FallbackChain([_CapturingAdapter([
            LLMResponse(content='{"intent":"general","entities":{},"sub_queries":[]}',
                        usage=_usage()),
            {"text": "购物助手回答", "tool_calls": None},
        ])], max_retries=0)
        monkeypatch.setattr(agent_mod, "get_fallback_llm",
                            lambda *a, **kw: FallbackChatModel(chain=chain))

        _agent_graph.invoke(
            {"messages": [{"role": "user", "content": "综合分析"}],
             "session_id": "force-2", "check_mode": "relaxed",
             "max_tool_rounds": 2, "force_analysis": False},
            config={"configurable": {"thread_id": "force-2"}},
        )
        content = "\n".join(
            str(getattr(m, "content", m)) for m in captured.get("decide_messages", [])
        )
        assert "购物助手" in content


class TestStreamEvents:
    def test_stream_event_sequence(self, monkeypatch):
        """SSE 流式事件序:meta → node_start/tool_call → answer → done(真实桥接+messages 模式)。"""
        import agent.agent as agent_mod
        monkeypatch.setattr(
            agent_mod, "SkillRegistry_get",
            lambda name: _fake_skill(SEG_STATS_DATA),
        )
        model = _llm_with_script([
            LLMResponse(content='{"intent":"user_segment","entities":{},"sub_queries":["各分群人数"]}',
                        usage=_usage()),
            {"text": "", "tool_calls": [
                {"name": "get_user_segment_stats", "args": {}, "id": "c1"},
            ]},
            LLMResponse(content='{"is_complete":true,"has_conflict":false,"missing_info":[],'
                                '"conflict_details":"","suggestion":"","confidence":0.9}',
                        usage=_usage()),
            {"text": "分群0有50人", "tool_calls": None},
        ], monkeypatch)

        from agent.agent import _ask_agent_stream

        async def _collect():
            events = []
            async for ev in _ask_agent_stream("各分群的人数是多少？", session_id="stream-test-1"):
                events.append(ev)
            return events

        events = asyncio.run(_collect())
        kinds = [e["event"] for e in events]
        assert kinds[0] == "meta"
        assert "node_start" in kinds
        assert "tool_call" in kinds
        assert "delta" in kinds                 # 真实桥接 → messages 模式捕获到 token 增量
        assert "answer" in kinds
        assert kinds[-1] == "done"
        done = events[-1]
        assert done["tools_called"] == ["get_user_segment_stats"]
