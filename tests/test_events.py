"""Unit tests for watcher/events.py — diff, detection, merge, virtual query."""

import uuid
from datetime import datetime, timezone

import pytest
from watcher.events import (
    SnapshotDiff,
    WatcherEvent,
    EventType,
    Priority,
    compute_snapshot_diff,
    detect_events,
    merge_low_priority,
    should_suppress,
    build_virtual_query,
    _finalize_group,
)
from pipeline.user_segmentation import ClusterSnapshot


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

NOW = datetime.now(timezone.utc).isoformat()
T1 = "2026-07-20T00:00:00"
T2 = "2026-07-25T00:00:00"


def _make_assignments(seg_stats: dict) -> dict:
    """Generate synthetic user_assignments matching the segment_stats totals."""
    assignments = {}
    uid = 1
    for seg, stats in seg_stats.items():
        for _ in range(stats["user_count"]):
            assignments[uid] = seg
            uid += 1
    return assignments


def _snapshot(timestamp: str, seg_stats: dict, flow_dist: dict | None = None,
              user_assignments: dict | None = None) -> ClusterSnapshot:
    if user_assignments is None:
        user_assignments = _make_assignments(seg_stats)
    return ClusterSnapshot(
        timestamp=timestamp,
        k_value=4,
        silhouette=0.65,
        segment_stats=seg_stats,
        user_assignments=user_assignments,
        flow_distribution=flow_dist or {},
    )


def _seg_stats(counts: list[tuple[int, int, int, float]]) -> dict:
    """Build segment_stats from (seg, count, avg_recency, avg_monetary) tuples."""
    return {
        seg: {
            "user_count": count,
            "avg_recency": float(rec),
            "avg_frequency": 3.0,
            "avg_monetary": float(mon),
        }
        for seg, count, rec, mon in counts
    }


def _flow_dist(seg_flows: dict[int, dict[str, int]]) -> dict:
    return seg_flows


# ══════════════════════════════════════════════════════════════════
# Diff Computation
# ══════════════════════════════════════════════════════════════════

class TestSnapshotDiff:
    def test_basic_diff(self):
        # 50 users in seg 0 + 100 in seg 1 = 150 total
        prev = _snapshot(T1, _seg_stats([(0, 50, 10, 100), (1, 100, 20, 200)]))
        curr = _snapshot(T2, _seg_stats([(0, 60, 12, 110), (1, 90, 18, 190)]))

        diff = compute_snapshot_diff(prev, curr)
        assert diff.prev_total_users == 150  # 50 + 100
        assert diff.curr_total_users == 150  # 60 + 90
        assert diff.total_users_delta == 0
        assert diff.segment_deltas[0]["user_count_delta"] == 10
        assert diff.segment_deltas[1]["user_count_delta"] == -10

    def test_zero_prev_users(self):
        prev = _snapshot(T1, {})
        curr = _snapshot(T2, _seg_stats([(0, 10, 5, 50)]))
        diff = compute_snapshot_diff(prev, curr)
        assert diff.total_users_pct_change == 100.0

    def test_zero_both(self):
        prev = _snapshot(T1, {})
        curr = _snapshot(T2, {})
        diff = compute_snapshot_diff(prev, curr)
        assert diff.total_users_pct_change == 0.0

    def test_flow_deltas(self):
        prev = _snapshot(T1, _seg_stats([(0, 50, 10, 100)]),
                         _flow_dist({0: {"active": 20, "churned": 5}}))
        curr = _snapshot(T2, _seg_stats([(0, 50, 10, 100)]),
                         _flow_dist({0: {"active": 25, "churned": 15}}))
        diff = compute_snapshot_diff(prev, curr)
        assert diff.flow_deltas[0]["active_delta"] == 5
        assert diff.flow_deltas[0]["churned_delta"] == 10


# ══════════════════════════════════════════════════════════════════
# Event Detection
# ══════════════════════════════════════════════════════════════════

class TestDetectEvents:
    def test_no_events_for_stable_data(self):
        prev = _snapshot(T1, _seg_stats([(0, 50, 10, 100), (1, 50, 20, 200)]))
        curr = _snapshot(T2, _seg_stats([(0, 50, 10, 100), (1, 50, 20, 200)]))
        diff = compute_snapshot_diff(prev, curr)
        events = detect_events(diff, prev, curr)
        assert len(events) == 0

    def test_order_volume_crash_detected(self):
        """30%+ user drop should trigger ORDER_VOLUME_CRASH."""
        prev = _snapshot(T1, _seg_stats([(0, 100, 10, 100)]))
        curr = _snapshot(T2, _seg_stats([(0, 60, 10, 100)]))
        diff = compute_snapshot_diff(prev, curr)
        events = detect_events(diff, prev, curr)
        types = [e.event_type for e in events]
        assert EventType.ORDER_VOLUME_CRASH in types

    def test_new_user_spike_detected(self):
        """20%+ user increase should trigger NEW_USER_SPIKE."""
        prev = _snapshot(T1, _seg_stats([(0, 100, 10, 100)]))
        curr = _snapshot(T2, _seg_stats([(0, 200, 10, 100)]))
        diff = compute_snapshot_diff(prev, curr)
        events = detect_events(diff, prev, curr)
        types = [e.event_type for e in events]
        assert EventType.NEW_USER_SPIKE in types

    def test_high_value_churn_detected(self):
        prev = _snapshot(T1,
                         _seg_stats([(0, 50, 10, 100), (2, 20, 5, 500)]),
                         _flow_dist({0: {"churned": 0}, 2: {"churned": 0}}))
        curr = _snapshot(T2,
                         _seg_stats([(0, 50, 10, 100), (2, 20, 5, 500)]),
                         _flow_dist({0: {"churned": 0}, 2: {"churned": 10}}))
        diff = compute_snapshot_diff(prev, curr)
        events = detect_events(diff, prev, curr)
        types = [e.event_type for e in events]
        assert EventType.HIGH_VALUE_CHURN in types


# ══════════════════════════════════════════════════════════════════
# Suppression
# ══════════════════════════════════════════════════════════════════

class TestSuppress:
    def test_high_never_suppressed(self):
        event = WatcherEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.HIGH_VALUE_CHURN,
            priority=Priority.HIGH,
            source_snapshot_ts=T2, prev_snapshot_ts=T1,
            details={}, virtual_query="", detected_at=NOW,
        )
        # None task manager → always False for HIGH
        assert should_suppress(event, None, 3600) is False


# ══════════════════════════════════════════════════════════════════
# Merge Window
# ══════════════════════════════════════════════════════════════════

class TestMerge:
    def _make_event(self, event_type: EventType, priority: Priority,
                    detected_at: str) -> WatcherEvent:
        return WatcherEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            priority=priority,
            source_snapshot_ts=T2, prev_snapshot_ts=T1,
            details={},
            virtual_query=f"Query for {event_type.value}",
            detected_at=detected_at,
        )

    def test_single_event_passes_through(self):
        e = self._make_event(EventType.SEGMENT_SHIFT, Priority.NORMAL, NOW)
        result = merge_low_priority([e], 600)
        assert len(result) == 1

    def test_high_events_preserved(self):
        e1 = self._make_event(EventType.HIGH_VALUE_CHURN, Priority.HIGH, NOW)
        e2 = self._make_event(EventType.SEGMENT_SHIFT, Priority.NORMAL, NOW)
        result = merge_low_priority([e1, e2], 600)
        assert len(result) == 2  # HIGH kept separate, NORMAL by itself

    def test_normals_within_window_merged(self):
        e1 = self._make_event(EventType.SEGMENT_SHIFT, Priority.NORMAL,
                              "2026-07-25T00:00:00")
        e2 = self._make_event(EventType.AVG_ORDER_CHANGE, Priority.NORMAL,
                              "2026-07-25T00:05:00")  # 5 min apart < 10 min window
        result = merge_low_priority([e1, e2], 600)
        assert len(result) == 1  # merged
        assert result[0].event_type == EventType.MERGED_LOW_PRIORITY

    def test_normals_outside_window_separated(self):
        e1 = self._make_event(EventType.SEGMENT_SHIFT, Priority.NORMAL,
                              "2026-07-25T00:00:00")
        e2 = self._make_event(EventType.AVG_ORDER_CHANGE, Priority.NORMAL,
                              "2026-07-25T01:00:00")  # 60 min apart > 10 min
        result = merge_low_priority([e1, e2], 600)
        assert len(result) == 2  # kept separate

    def test_merge_window_uses_last_event(self):
        """Regression: must compare against group[-1], not group[0]."""
        e1 = self._make_event(EventType.SEGMENT_SHIFT, Priority.NORMAL,
                              "2026-07-25T00:00:00")
        e2 = self._make_event(EventType.AVG_ORDER_CHANGE, Priority.NORMAL,
                              "2026-07-25T00:05:00")   # 5 min after e1 → merged
        e3 = self._make_event(EventType.NEW_USER_SPIKE, Priority.NORMAL,
                              "2026-07-25T00:12:00")   # 7 min after e2 → merged (within window)
        # Old bug: compared e3 against e1 (12 min gap → not merged)
        # Fixed:  compares e3 against e2 (7 min gap → merged)
        result = merge_low_priority([e1, e2, e3], 600)
        assert len(result) == 1  # All three should be merged


# ══════════════════════════════════════════════════════════════════
# Virtual Query Rendering
# ══════════════════════════════════════════════════════════════════

class TestVirtualQuery:
    def test_churn_template_rendered(self):
        details = {
            "top_segment": 2, "churn_count": 10,
            "churn_pct": 50.0, "prev_churned": 0, "curr_churned": 10,
        }
        q = build_virtual_query(EventType.HIGH_VALUE_CHURN, details, T1, T2)
        assert "高价值用户流失" in q
        assert "segment 2" in q or "2" in q

    def test_unknown_type_fallback(self):
        q = build_virtual_query(EventType.MERGED_LOW_PRIORITY, {}, T1, T2)
        assert len(q) > 0  # should not crash


# ══════════════════════════════════════════════════════════════════
# Finalize Group
# ══════════════════════════════════════════════════════════════════

class TestFinalizeGroup:
    def test_single_event_unchanged(self):
        e = WatcherEvent(
            event_id="single", event_type=EventType.SEGMENT_SHIFT,
            priority=Priority.NORMAL,
            source_snapshot_ts=T2, prev_snapshot_ts=T1,
            details={}, virtual_query="original query", detected_at=NOW,
        )
        result = _finalize_group([e])
        assert result.event_type == EventType.SEGMENT_SHIFT
        assert result.virtual_query == "original query"

    def test_multiple_merged(self):
        e1 = WatcherEvent(
            event_id="e1", event_type=EventType.SEGMENT_SHIFT,
            priority=Priority.NORMAL,
            source_snapshot_ts=T2, prev_snapshot_ts=T1,
            details={}, virtual_query="query 1", detected_at=NOW,
        )
        e2 = WatcherEvent(
            event_id="e2", event_type=EventType.AVG_ORDER_CHANGE,
            priority=Priority.NORMAL,
            source_snapshot_ts=T2, prev_snapshot_ts=T1,
            details={}, virtual_query="query 2", detected_at=NOW,
        )
        result = _finalize_group([e1, e2])
        assert result.event_type == EventType.MERGED_LOW_PRIORITY
        assert "query 1" in result.virtual_query
        assert "query 2" in result.virtual_query
        assert result.details["merged_event_ids"] == ["e1", "e2"]
