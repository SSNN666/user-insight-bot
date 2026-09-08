"""会话持久化测试:JSONCheckpointSaver 的 put/get/TTL + 图重启后历史恢复。"""
import os
import time

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from agent.session_store import JSONCheckpointSaver, LazyCheckpointSaver


def _checkpoint(msgs: list) -> dict:
    return {
        "v": 1,
        "id": "ckpt-1",
        "ts": "2026-08-16T00:00:00Z",
        "channel_values": {"messages": msgs},
        "channel_versions": {"messages": "v1"},
        "versions_seen": {},
        "pending_sends": [],
    }


class TestSaver:
    def test_put_get_roundtrip(self, tmp_path):
        saver = JSONCheckpointSaver(str(tmp_path))
        config = {"configurable": {"thread_id": "s1"}}
        msgs = [HumanMessage(content="你好"), AIMessage(content="你好,我是助手")]
        saver.put(config, _checkpoint(msgs), {}, {})

        tup = saver.get_tuple(config)
        assert tup is not None
        loaded = tup.checkpoint["channel_values"]["messages"]
        assert [m.content for m in loaded] == ["你好", "你好,我是助手"]

    def test_get_missing_returns_none(self, tmp_path):
        saver = JSONCheckpointSaver(str(tmp_path))
        assert saver.get_tuple({"configurable": {"thread_id": "nope"}}) is None

    def test_delete_thread(self, tmp_path):
        saver = JSONCheckpointSaver(str(tmp_path))
        config = {"configurable": {"thread_id": "s1"}}
        saver.put(config, _checkpoint([HumanMessage(content="x")]), {}, {})
        saver.delete_thread("s1")
        assert saver.get_tuple(config) is None

    def test_ttl_sweep(self, tmp_path):
        saver = JSONCheckpointSaver(str(tmp_path), ttl_days=7)
        saver.put({"configurable": {"thread_id": "old"}},
                  _checkpoint([HumanMessage(content="x")]), {}, {})
        fpath = saver._path("old")
        old_time = time.time() - 8 * 86400   # 8 天前
        os.utime(fpath, (old_time, old_time))
        saver._last_sweep = 0.0   # 重置清理节流,触发立即扫描
        # 写入另一个线程触发惰性清理
        saver.put({"configurable": {"thread_id": "new"}},
                  _checkpoint([HumanMessage(content="y")]), {}, {})
        assert not os.path.isfile(fpath)   # 过期文件已清理
        assert saver.get_tuple({"configurable": {"thread_id": "new"}}) is not None


class TestGraphRestart:
    def test_history_survives_graph_restart(self, tmp_path, monkeypatch):
        """新图实例 + 同一 saver + 同一 thread_id → 多轮历史自动恢复。"""
        import tests.test_agent_graph as tg
        from agent.agent import _build_graph
        from llm.adapter import LLMResponse, UsageInfo

        saver = JSONCheckpointSaver(str(tmp_path))
        graph1 = _build_graph(checkpointer=saver)
        graph2 = _build_graph(checkpointer=saver)

        def _usage():
            return UsageInfo(total_tokens=5, provider="scripted")

        # 第一轮(图1):preprocess → decide 直接回答
        model1 = tg._llm_with_script([
            LLMResponse(content='{"intent":"general","entities":{},"sub_queries":["你好"]}',
                        usage=_usage()),
            {"text": "你好,我是助手", "tool_calls": None},
        ], monkeypatch)
        r1 = graph1.invoke(
            {"messages": [{"role": "user", "content": "你好"}],
             "session_id": "restart-1", "check_mode": "relaxed",
             "max_tool_rounds": 2},
            config={"configurable": {"thread_id": "restart-1"}},
        )
        assert r1["messages"]

        # 第二轮(图2 新实例):脚本续接,历史应含第一轮消息
        tg._llm_with_script([
            LLMResponse(content='{"intent":"general","entities":{},"sub_queries":["再见"]}',
                        usage=_usage()),
            {"text": "再见!", "tool_calls": None},
        ], monkeypatch)
        r2 = graph2.invoke(
            {"messages": [{"role": "user", "content": "再见"}],
             "session_id": "restart-1", "check_mode": "relaxed",
             "max_tool_rounds": 2},
            config={"configurable": {"thread_id": "restart-1"}},
        )
        contents = [getattr(m, "content", "") for m in r2["messages"]]
        assert "你好" in contents          # 第一轮用户消息仍在
        assert "你好,我是助手" in contents   # 第一轮 AI 回复仍在
        assert "再见!" in contents


# ── 双轨:postgres 模式选择与 fail-fast(离线,不依赖真实 PG)──────────


class TestStoreDualTrack:
    def test_default_store_is_json(self):
        """SESSION_STORE 默认 json:代理懒建 JSON saver。"""
        saver = LazyCheckpointSaver()
        assert isinstance(saver._ensure(), JSONCheckpointSaver)

    def test_json_warmup_ok(self):
        """json 模式 warmup 不抛错(预创建会话目录)。"""
        LazyCheckpointSaver().warmup()

    def test_postgres_mode_without_dsn_fails_fast(self, monkeypatch):
        """SESSION_STORE=postgres 但 DSN 缺失 → 启动即抛,不静默降级。"""
        monkeypatch.setenv("SESSION_STORE", "postgres")
        monkeypatch.setenv("SESSION_POSTGRES_DSN", "")
        saver = LazyCheckpointSaver()
        with pytest.raises(RuntimeError, match="未配置 SESSION_POSTGRES_DSN"):
            saver.warmup()

    def test_postgres_mode_unreachable_dsn_fails_fast(self, monkeypatch):
        """DSN 连不上 → fail-fast RuntimeError(端口 1 必然拒绝,毫秒级)。"""
        monkeypatch.setenv("SESSION_STORE", "postgres")
        monkeypatch.setenv(
            "SESSION_POSTGRES_DSN",
            "postgresql://nobody:nope@127.0.0.1:1/nodb?connect_timeout=2",
        )
        saver = LazyCheckpointSaver()
        with pytest.raises(RuntimeError, match="SESSION_STORE=postgres 连接失败"):
            saver.warmup()
