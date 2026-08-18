"""自动周报:调度时刻计算 + 报告生成(幂等/降级)+ 事件时间窗口查询。

全部离线:不调 LLM,快照用 save_snapshot 合成,任务用 TaskManager 直插。
"""
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from watcher.weekly_report import (
    _iso_week_str, _next_report_time, _tz, generate_weekly_report,
    list_recent_reports,
)
from watcher.events import EventType, Priority, WatcherEvent
from watcher.task_manager import TaskManager
from pipeline.user_segmentation import save_snapshot
from config.settings import get_settings

UTC = timezone.utc
TZ = _tz()


@pytest.fixture(autouse=True)
def _fresh_task_manager():
    """generate_weekly_report 内部用 get_task_manager() 单例 —— 每测试重置,
    保证读到 conftest 重定向后的 WATCHER_DB_PATH,且互不污染。"""
    import watcher.task_manager as tm_mod
    tm_mod._task_manager = None
    yield
    tm_mod._task_manager = None


@pytest.fixture(autouse=True)
def _no_real_pipeline(monkeypatch):
    """get_segment_names() 内部会触发全量流水线(读真实 JData + 污染快照目录)。

    segment_naming 用函数内 from-import(每次调用重新绑定)→ monkeypatch
    源模块属性即可隔离,命名走确定性启发式(零 LLM 依赖)。
    """
    seg_df = pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6],
        "recency": [3, 10, 20, 30, 45, 60],
        "frequency": [5.0, 3.0, 2.0, 1.5, 1.0, 1.0],
        "monetary": [800.0, 500.0, 300.0, 200.0, 100.0, 80.0],
        "segment": [2, 2, 1, 1, 0, 0],
        "flow_tag": ["active", "active", "potential", "dormant", "churned", "churned"],
    })
    monkeypatch.setattr(
        "skills.user_segment._load_and_process",
        lambda force_refresh=False: (None, seg_df, None))
    yield


def _make_rfm() -> pd.DataFrame:
    """6 用户 3 分群(max=2):高价值分群 2 人,含 active/potential。"""
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4, 5, 6],
        "recency": [3, 10, 20, 30, 45, 60],
        "frequency": [5.0, 3.0, 2.0, 1.5, 1.0, 1.0],
        "monetary": [800.0, 500.0, 300.0, 200.0, 100.0, 80.0],
        "segment": [2, 2, 1, 1, 0, 0],
        "flow_tag": ["active", "active", "potential", "dormant", "churned", "churned"],
    })


def _seed_task(db_path: str, event_type: str = "segment_shift",
               query: str = "分群1人数变化5%,请分析原因") -> TaskManager:
    tm = TaskManager(db_path)
    ev = WatcherEvent(
        event_id="test-1", event_type=EventType(event_type),
        priority=Priority.NORMAL, source_snapshot_ts="t", prev_snapshot_ts="",
        details={"seg_id": 1}, virtual_query=query,
        detected_at=datetime.now(UTC).isoformat(),
    )
    tm.create_task(ev)
    return tm


# ══════════════════════════════════════════════════════════════════
# 调度时刻(纯函数)
# ══════════════════════════════════════════════════════════════════

class TestNextReportTime:
    def test_wednesday_targets_next_monday(self):
        now = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)          # 周三
        t = _next_report_time(now, 9, TZ)
        assert t.weekday() == 0                                 # 周一
        assert t.hour == 9
        assert t.date().isoformat() == "2026-08-24"

    def test_sunday_late_targets_next_monday(self):
        now = datetime(2026, 8, 23, 23, 59, tzinfo=UTC)         # 周日晚
        t = _next_report_time(now, 9, TZ)
        assert t.date().isoformat() == "2026-08-24" and t.hour == 9

    def test_monday_before_hour_targets_same_day(self):
        now = datetime(2026, 8, 17, 0, 30, tzinfo=UTC)          # 周一凌晨
        t = _next_report_time(now, 9, TZ)
        assert t.date().isoformat() == "2026-08-17" and t.hour == 9

    def test_monday_after_hour_targets_next_week(self):
        now = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)          # 周一上午(已过 9 点)
        t = _next_report_time(now, 9, TZ)
        assert t.date().isoformat() == "2026-08-24"

    def test_iso_week_str(self):
        assert _iso_week_str(datetime(2026, 8, 17, tzinfo=UTC)) == "2026-W34"


# ══════════════════════════════════════════════════════════════════
# generate_weekly_report(幂等/降级/落盘)
# ══════════════════════════════════════════════════════════════════

class TestGenerateWeeklyReport:
    def test_report_content_and_persistence(self):
        settings = get_settings()
        # 快照 ×2 + 事件 ×1
        rfm = _make_rfm()
        save_snapshot(rfm, k_value=3, silhouette=0.6)
        save_snapshot(rfm, k_value=3, silhouette=0.6)
        tm = _seed_task(os.path.join(settings.WATCHER_DB_PATH))

        result = generate_weekly_report()                       # now=None → 真实当前周
        assert result["generated"] is True
        assert result["week"].startswith("20")                  # 2026-Wxx
        assert os.path.exists(result["path"])

        content = result["content"]
        for section in ["# 📊 运营周报", "## 一、本周自动监测事件",
                        "## 二、分群规模与趋势", "## 三、高价值分群重点",
                        "## 四、运营建议"]:
            assert section in content
        assert "segment_shift" in content                        # 事件表格行
        assert "分群1人数变化5%" in content                       # 事件摘要
        assert "高价值" in content and "segment 2" in content     # 高价值分群小节

    def test_idempotent_same_week(self):
        settings = get_settings()
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        r1 = generate_weekly_report()
        r2 = generate_weekly_report()
        assert r1["generated"] is True and r2["generated"] is False
        assert r1["content"] == r2["content"]

    def test_force_regenerates(self):
        settings = get_settings()
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        r1 = generate_weekly_report()
        r2 = generate_weekly_report(force=True)
        assert r2["generated"] is True

    def test_insufficient_snapshots_degraded(self):
        """窗口内快照 <2 → 降级使用最近两期;只有一个快照 → 明确标注。"""
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)    # 只有 1 个
        result = generate_weekly_report()
        assert "快照不足" in result["content"]

    def test_no_events_section_still_generated(self):
        settings = get_settings()
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        result = generate_weekly_report()
        assert "本周无自动监测事件" in result["content"]

    def test_list_recent_reports(self):
        settings = get_settings()
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        save_snapshot(_make_rfm(), k_value=3, silhouette=0.6)
        r = generate_weekly_report()
        recent = list_recent_reports(limit=1)
        assert recent and recent[0]["week"] == r["week"]
        assert recent[0]["content"] == r["content"]
        assert list_recent_reports(limit=0) == []


# ══════════════════════════════════════════════════════════════════
# list_tasks_by_time(周报数据源查询)
# ══════════════════════════════════════════════════════════════════

class TestListTasksByTime:
    def test_time_window_and_type_filter(self, tmp_path):
        db = os.path.join(str(tmp_path), "watcher.db")
        tm = _seed_task(db, event_type="segment_shift", query="窗口内事件")
        tm.create_task(WatcherEvent(
            event_id="t2", event_type=EventType.NEW_USER_SPIKE,
            priority=Priority.NORMAL, source_snapshot_ts="t", prev_snapshot_ts="",
            details={}, virtual_query="另一个事件",
            detected_at=datetime.now(UTC).isoformat(),
        ))

        now = datetime.now(UTC)
        since = (now - timedelta(hours=1)).isoformat()
        until = (now + timedelta(hours=1)).isoformat()

        all_tasks = tm.list_tasks_by_time(since, until, limit=50)
        assert len(all_tasks) == 2
        # 类型过滤
        seg_tasks = tm.list_tasks_by_time(since, until, event_type="segment_shift")
        assert len(seg_tasks) == 1 and seg_tasks[0].event_type == "segment_shift"
        # 未来窗口 → 空
        future_since = (now + timedelta(days=2)).isoformat()
        assert tm.list_tasks_by_time(future_since, until) == []
