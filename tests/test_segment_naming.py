"""分群业务命名:启发式兜底 + LLM 路径 + 缓存,全部离线。"""
import pandas as pd
import pytest

import pipeline.segment_naming as sn

SEG = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6],
    "segment": [0, 0, 1, 1, 2, 2],
    "recency": [50.0, 43.8, 1.5, 1.1, 40.0, 34.3],
    "frequency": [1, 1, 1, 1, 3, 3],
    "monetary": [2478.82, 2478.82, 2591.91, 2591.91, 7678.67, 7678.67],
    "flow_tag": ["stable", "potential", "active", "active", "potential", "potential"],
})


class TestHeuristicNames:
    def test_value_tier_ordering(self):
        names = sn.heuristic_names(SEG)
        assert names[0].startswith("低价值")
        assert names[1].startswith("中坚") or names[1].startswith("低价值")
        assert names[2].startswith("高价值")

    def test_flow_dominant_used(self):
        names = sn.heuristic_names(SEG)
        assert "潜力" in names[2]          # 分群2 主流转 = potential

    def test_empty_seg(self):
        assert sn.heuristic_names(pd.DataFrame()) == {}


class TestLlmNames:
    def _patch(self, monkeypatch, text: str):
        monkeypatch.setattr(sn, "_call_naming_llm", lambda ctx: text)

    def test_valid_json_names(self, monkeypatch):
        self._patch(monkeypatch, '{"0": "低活长尾用户", "1": "新晋活跃用户", "2": "高价值核心用户"}')
        names = sn._llm_names(SEG)
        assert names == {0: "低活长尾用户", 1: "新晋活跃用户", 2: "高价值核心用户"}

    def test_json_with_fences_and_noise(self, monkeypatch):
        """LLM 输出带 markdown fence 与前后废话 → json_repair 容错。"""
        self._patch(monkeypatch,
                    '好的,如下:\n```json\n{"0": "低活长尾用户", "1": "新晋活跃用户", "2": "高价值核心用户"}\n```')
        names = sn._llm_names(SEG)
        assert names is not None and names[2] == "高价值核心用户"

    def test_wrong_keys_rejected(self, monkeypatch):
        """键与真实分群不符 → None → 走启发式。"""
        self._patch(monkeypatch, '{"0": "A用户", "9": "B用户"}')
        assert sn._llm_names(SEG) is None

    def test_missing_one_segment_rejected(self, monkeypatch):
        self._patch(monkeypatch, '{"0": "A用户", "1": "B用户"}')
        assert sn._llm_names(SEG) is None

    def test_garbage_output_rejected(self, monkeypatch):
        self._patch(monkeypatch, "抱歉我无法完成")
        assert sn._llm_names(SEG) is None


class TestGetSegmentNames:
    def test_llm_path_wins_and_caches(self, monkeypatch):
        monkeypatch.setattr(
            sn, "_call_naming_llm",
            lambda ctx: '{"0": "低活长尾", "1": "新晋活跃", "2": "高价值核心"}',
        )
        sn.invalidate_segment_names()
        names1 = sn.get_segment_names(SEG)
        assert names1[2] == "高价值核心"

        # 第二次命中缓存,即使 LLM 挂掉也返回同样的名字
        monkeypatch.setattr(sn, "_call_naming_llm", lambda ctx: (_ for _ in ()).throw(RuntimeError()))
        names2 = sn.get_segment_names(SEG)
        assert names2 == names1

    def test_llm_down_falls_back_heuristic(self, monkeypatch):
        def _boom(ctx):
            raise RuntimeError("all providers down")
        monkeypatch.setattr(sn, "_call_naming_llm", _boom)
        sn.invalidate_segment_names()
        names = sn.get_segment_names(SEG)
        assert names[2].startswith("高价值")     # 启发式兜底仍可用
