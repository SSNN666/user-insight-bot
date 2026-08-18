"""流失预警:detect_high_value_dormant 纯函数 + dormant_state 持久化 + 冷却语义。

全部离线,不触 LLM/不跑流水线(合成 rfm DataFrame)。
"""
import os

import pandas as pd
import pytest

from watcher import events, dormant_state
from watcher.events import (
    EventType, Priority, WatcherEvent, detect_high_value_dormant, should_suppress,
)
from config.settings import get_settings


def _rfm_df(recencies: list[float], segment: int = 2) -> pd.DataFrame:
    """合成 rfm:所有用户同一分群(segment),recency 可指定。"""
    return pd.DataFrame({
        "user_id": list(range(1, len(recencies) + 1)),
        "recency": recencies,
        "segment": [segment] * len(recencies),
        "frequency": [3.0] * len(recencies),
        "monetary": [500.0] * len(recencies),
    })


def _settings():
    return get_settings()


# ══════════════════════════════════════════════════════════════════
# detect_high_value_dormant(纯函数)
# ══════════════════════════════════════════════════════════════════

class TestDetectHighValueDormant:
    def test_high_event_when_threshold_met(self):
        rfm = _rfm_df([31, 35, 40, 45, 50, 10, 5])   # 5 人 > 30 天,2 人活跃
        ev = detect_high_value_dormant(rfm, _settings())
        assert ev is not None
        assert ev.event_type == EventType.HIGH_VALUE_DORMANT
        assert ev.priority == Priority.HIGH
        assert ev.details["top_segment"] == 2
        assert ev.details["new_dormant_count"] == 5
        assert ev.details["new_dormant_ids"] == [1, 2, 3, 4, 5]
        assert ev.details["sample_ids"] == [1, 2, 3, 4, 5]
        assert ev.details["total_dormant_in_seg"] == 5
        # virtual_query 走模板:含流失人数与召回策略要求
        assert "5" in ev.virtual_query and "召回" in ev.virtual_query

    def test_below_min_users_returns_none(self):
        rfm = _rfm_df([31, 35, 40])                 # 3 人 < MIN_USERS=5
        assert detect_high_value_dormant(rfm, _settings()) is None

    def test_recency_within_threshold_returns_none(self):
        rfm = _rfm_df([5, 10, 20, 25, 29, 30, 15])  # 全部 ≤ 30 天
        assert detect_high_value_dormant(rfm, _settings()) is None

    def test_prev_reported_only_counts_new(self):
        rfm = _rfm_df([31, 35, 40, 45, 50, 55, 60, 65])
        ev = detect_high_value_dormant(rfm, _settings(), prev_reported={1, 2, 3})
        assert ev is not None
        assert ev.details["new_dormant_ids"] == [4, 5, 6, 7, 8]   # 只计新增
        assert ev.details["new_dormant_count"] == 5

    def test_all_reported_returns_none(self):
        rfm = _rfm_df([31, 35, 40, 45, 50])
        assert detect_high_value_dormant(rfm, _settings(), prev_reported={1, 2, 3, 4, 5}) is None

    def test_empty_rfm_returns_none(self):
        assert detect_high_value_dormant(None, _settings()) is None
        assert detect_high_value_dormant(
            pd.DataFrame(columns=["user_id", "recency", "segment"]), _settings()) is None

    def test_missing_columns_returns_none(self):
        bad = pd.DataFrame({"user_id": [1, 2], "recency": [31, 35]})   # 无 segment
        assert detect_high_value_dormant(bad, _settings()) is None

    def test_max_segment_respected(self):
        # segment 1 = max:只有 segment 1 的用户参与判定
        rfm = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6, 7],
            "recency": [31, 35, 40, 45, 50, 10, 5],
            "segment": [2, 2, 2, 2, 2, 1, 1],       # segment 2 是最大分群
        })
        ev = detect_high_value_dormant(rfm, _settings())
        assert ev is not None
        assert ev.details["top_segment"] == 2
        assert ev.details["new_dormant_ids"] == [1, 2, 3, 4, 5]


# ══════════════════════════════════════════════════════════════════
# dormant_state(持久化)
# ══════════════════════════════════════════════════════════════════

class TestDormantState:
    def test_save_load_roundtrip(self):
        dormant_state.mark_reported([1, 2, 3])
        dormant_state.mark_reported([3, 4])          # 合并
        assert dormant_state.load_reported() == {1, 2, 3, 4}

    def test_missing_file_returns_empty(self):
        assert dormant_state.load_reported() == set()

    def test_corrupt_file_returns_empty(self):
        # 写坏文件 → 读回空集,不阻塞主链路
        settings = get_settings()
        path = os.path.join(settings.CACHE_DIR, "dormant_reported.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        assert dormant_state.load_reported() == set()


# ══════════════════════════════════════════════════════════════════
# 冷却语义(次闸)+ should_suppress 语义锁定
# ══════════════════════════════════════════════════════════════════

class TestDormantCooldown:
    def _make_event(self) -> WatcherEvent:
        ev = detect_high_value_dormant(_rfm_df([31, 35, 40, 45, 50]), _settings())
        assert ev is not None
        return ev

    def test_cooldown_blocks_second_trigger(self, tmp_path):
        from watcher.task_manager import TaskManager
        tm = TaskManager(os.path.join(str(tmp_path), "watcher.db"))
        ev = self._make_event()
        tm.create_task(ev)
        assert tm.is_event_in_cooldown(
            "high_value_dormant", 86400) is True     # 冷却期内 → 次闸拦截

    def test_no_recent_task_no_cooldown(self, tmp_path):
        from watcher.task_manager import TaskManager
        tm = TaskManager(os.path.join(str(tmp_path), "watcher.db"))
        assert tm.is_event_in_cooldown("high_value_dormant", 86400) is False

    def test_should_suppress_semantics_unchanged(self):
        """HIGH 事件永不抑制 —— 锁定不改 should_suppress 语义(有既有测试守护)。"""
        ev = self._make_event()
        assert ev.priority == Priority.HIGH
        assert should_suppress(ev, None, 3600) is False
