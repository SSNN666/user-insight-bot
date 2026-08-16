"""环比/趋势 Skill:快照周期选择 + 指定月份路径 + 无快照降级,全部离线。"""
import pytest

from skills.user_segment import (
    _pick_snapshot_periods, SegmentGrowthSkill, SegmentTrendSkill,
)
from pipeline.user_segmentation import ClusterSnapshot


def _snap(ts: str, stats: dict) -> ClusterSnapshot:
    return ClusterSnapshot(
        timestamp=ts, k_value=3, silhouette=0.5,
        segment_stats=stats, user_assignments={}, flow_distribution={},
    )


def _stats(n0, m0, n2, m2):
    return {
        0: {"user_count": n0, "avg_recency": 40.0, "avg_frequency": 1.1, "avg_monetary": m0},
        2: {"user_count": n2, "avg_recency": 37.0, "avg_frequency": 2.8, "avg_monetary": m2},
    }


SNAPS = [
    _snap("2026-07-26T19:45:00", _stats(60, 5100, 20, 7600)),
    _snap("2026-07-26T19:51:00", _stats(67, 5200, 23, 7700)),
    _snap("2026-08-16T13:11:00", _stats(70, 5300, 25, 7800)),
    _snap("2026-08-16T22:18:00", _stats(75, 5400, 28, 7900)),
]


class TestPickPeriods:
    def test_default_picks_last_two(self):
        s1, s2, err = _pick_snapshot_periods(SNAPS, "", "")
        assert err == ""
        assert s1.timestamp == "2026-08-16T13:11:00"
        assert s2.timestamp == "2026-08-16T22:18:00"

    def test_explicit_months_pick_last_of_each(self):
        s1, s2, err = _pick_snapshot_periods(SNAPS, "2026-07", "2026-08")
        assert err == ""
        assert s1.timestamp == "2026-07-26T19:51:00"      # 7 月最后一个
        assert s2.timestamp == "2026-08-16T22:18:00"      # 8 月最后一个

    def test_missing_month_returns_hint(self):
        s1, s2, err = _pick_snapshot_periods(SNAPS, "2016-02", "2016-03")
        assert s1 is None and s2 is None
        assert "无快照" in err and "2026-07" in err       # 提示现有月份

    def test_single_period_requires_both(self):
        s1, s2, err = _pick_snapshot_periods(SNAPS, "2026-07", "")
        assert s1 is None and s2 is None and "同时" in err


class TestSkillExecute:
    def test_no_snapshots_returns_partial(self, monkeypatch):
        import skills.user_segment as us
        monkeypatch.setattr(us, "_load_and_process",
                            lambda force_refresh=False: (None, None, None))
        monkeypatch.setattr("pipeline.user_segmentation.load_snapshots",
                            lambda **kw: [])
        r = SegmentGrowthSkill().execute()
        assert r.status.value == "partial"
        assert "至少2个快照" in r.summary

    def test_specified_periods_with_data(self, monkeypatch):
        """指定月份对比:数据行含人数/消费变化(来自 compare_snapshots)。"""
        import skills.user_segment as us
        monkeypatch.setattr(us, "_load_and_process",
                            lambda force_refresh=False: (None, None, None))
        monkeypatch.setattr("pipeline.user_segmentation.load_snapshots",
                            lambda **kw: SNAPS)
        monkeypatch.setattr("pipeline.segment_naming.get_segment_names",
                            lambda seg=None: {0: "低活长尾", 2: "高价值核心"})
        r = SegmentGrowthSkill().execute(period1="2026-07", period2="2026-08")
        assert r.status.value == "success"
        assert "2026-07-26 → 2026-08-16" in r.summary
        rows = {g["segment"]: g for g in r.data}
        assert rows[0]["user_count_delta"] == 8       # 67 → 75
        assert rows[2]["avg_monetary_delta"] == pytest.approx(200, 0.01)  # 7700 → 7900
        assert "低活长尾" in r.summary                  # 业务名进对比表


class TestTrendSkill:
    def _patch(self, monkeypatch):
        import skills.user_segment as us
        monkeypatch.setattr(us, "_load_and_process",
                            lambda force_refresh=False: (None, None, None))
        monkeypatch.setattr("pipeline.user_segmentation.load_snapshots",
                            lambda **kw: SNAPS)
        monkeypatch.setattr("pipeline.segment_naming.get_segment_names",
                            lambda seg=None: {0: "低活长尾", 2: "高价值核心"})

    def test_returns_timeseries_rows(self, monkeypatch):
        self._patch(monkeypatch)
        r = SegmentTrendSkill().execute()
        assert r.status.value == "success"
        assert {"timestamp", "segment", "user_count"} <= set(r.data[0].keys())
        assert "低活长尾" in r.summary
        assert "60 → 75" in r.summary                  # 首末快照人数对比

    def test_trend_rows_become_line_chart(self, monkeypatch):
        """趋势 Skill 的数据行 → chart_extract 折线图(对话内出图链路)。"""
        self._patch(monkeypatch)
        r = SegmentTrendSkill().execute()
        from agent.chart_extract import build_charts_from_skills
        charts = build_charts_from_skills([r], {0: "低活长尾", 2: "高价值核心"})
        assert len(charts) == 1
        c = charts[0]
        assert c["type"] == "line"
        assert len(c["categories"]) == 4               # 4 个快照时间点
        assert len(c["series"]) == 2                   # 分群 0 与 2
        assert "低活长尾" in c["series"][0]["name"]

    def test_insufficient_snapshots_partial(self, monkeypatch):
        import skills.user_segment as us
        monkeypatch.setattr(us, "_load_and_process",
                            lambda force_refresh=False: (None, None, None))
        monkeypatch.setattr("pipeline.user_segmentation.load_snapshots",
                            lambda **kw: SNAPS[:1])
        r = SegmentTrendSkill().execute()
        assert r.status.value == "partial"
        assert "快照不足" in r.summary
