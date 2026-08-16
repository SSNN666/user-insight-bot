"""分群运营建议:上下文构建 + 剧本化 LLM + 数值核查回退重试 + 端点接线。

全部离线:_call_llm 被剧本替换,不发真实 LLM 请求。
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import agent.suggestions as sug

# 与真实 JData 3 分群同口径的迷你分群表
SEG = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6],
    "segment": [0, 0, 1, 1, 2, 2],
    "recency": [50.0, 43.8, 1.5, 1.1, 40.0, 34.3],
    "frequency": [1, 1, 1, 1, 3, 3],
    "monetary": [2478.82, 2478.82, 2591.91, 2591.91, 7678.67, 7678.67],
    "flow_tag": ["stable", "stable", "active", "active", "potential", "potential"],
})


class TestContextBuilding:
    def test_stats_rows_chinese_keys(self):
        rows = sug._build_stats_rows(SEG)
        assert len(rows) == 3
        r0, r2 = rows[0], rows[2]
        assert r0["segment"] == 0 and r0["用户数"] == 2
        assert r0["平均消费"] == 2478.82
        assert r2["平均消费"] == 7678.67
        assert set(r0.keys()) == {"segment", "用户数", "平均近度", "平均频次", "平均消费"}

    def test_context_contains_real_numbers(self):
        ctx = sug.build_suggestion_context(SEG)
        assert "分群统计" in ctx and "流转分布" in ctx
        assert "7678.67" in ctx and "2478.82" in ctx
        assert '"segment": 2' in ctx


class TestGeneration:
    def _patch_llm(self, monkeypatch, replies: list[str]):
        queue = list(replies)
        monkeypatch.setattr(sug, "_call_llm", lambda messages: queue.pop(0))

    def test_grounded_text_passes(self, monkeypatch):
        self._patch_llm(monkeypatch, [
            "分群0有2人，平均消费为2478.82元。"
            "分群2的平均消费达7678.67元，建议提供专属权益。",
        ])
        result = sug.generate_segment_suggestions(seg=SEG)
        assert result["grounded"] is True
        assert result["violations"] == 0
        assert result["retries"] == 0

    def test_hallucination_retried_once_then_grounded(self, monkeypatch):
        """第一次编造 9999 元 → 附修正提示重生成 → 第二次通过。"""
        self._patch_llm(monkeypatch, [
            "分群2的平均消费为9999元，建议大促。",
            "分群2的平均消费为7678.67元，建议专属权益。",
        ])
        result = sug.generate_segment_suggestions(seg=SEG)
        assert result["grounded"] is True
        assert result["retries"] == 1

    def test_persistent_hallucination_marked_ungrounded(self, monkeypatch):
        self._patch_llm(monkeypatch, [
            "分群2的平均消费为9999元。",
            "分群2的平均消费为8888元。",
        ])
        result = sug.generate_segment_suggestions(seg=SEG)
        assert result["grounded"] is False
        assert result["violations"] >= 1
        assert result["retries"] == 1          # 重试上限用尽,如实标注


class TestEndpoint:
    def test_suggestions_endpoint(self, monkeypatch):
        monkeypatch.setattr(sug, "generate_segment_suggestions",
                            lambda: {"suggestions": "建议X", "grounded": True,
                                     "violations": 0, "retries": 0})
        import api.main as main_mod
        client = TestClient(main_mod.app)
        r = client.get("/api/suggestions")
        assert r.status_code == 200
        body = r.json()
        assert body["suggestions"] == "建议X"
        assert body["grounded"] is True
