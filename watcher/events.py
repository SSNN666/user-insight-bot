"""Event detection, deduplication, and virtual query generation.

Uses Phase 3's ClusterSnapshot + compare_snapshots() as data source.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from pipeline.user_segmentation import (
    ClusterSnapshot, load_snapshots, compare_snapshots,
)
from log.logger import get_logger

logger = get_logger(__name__)


# ── Enums ───────────────────────────────────────────────────────


class EventType(StrEnum):
    HIGH_VALUE_CHURN = "high_value_churn"
    HIGH_VALUE_DORMANT = "high_value_dormant"
    ORDER_VOLUME_CRASH = "order_volume_crash"
    EXTREME_OUTLIER = "extreme_outlier"
    SEGMENT_SHIFT = "segment_shift"
    AVG_ORDER_CHANGE = "avg_order_change"
    NEW_USER_SPIKE = "new_user_spike"
    MERGED_LOW_PRIORITY = "merged_low_priority"


class Priority(StrEnum):
    HIGH = "high"
    NORMAL = "normal"


# ── Data Models ─────────────────────────────────────────────────


@dataclass
class SnapshotDiff:
    prev_timestamp: str
    curr_timestamp: str
    prev_total_users: int
    curr_total_users: int
    total_users_delta: int
    total_users_pct_change: float
    segment_deltas: dict = field(default_factory=dict)
    flow_deltas: dict = field(default_factory=dict)
    avg_monetary_delta_pct: float = 0.0
    avg_frequency_delta_pct: float = 0.0


@dataclass
class WatcherEvent:
    event_id: str
    event_type: EventType
    priority: Priority
    source_snapshot_ts: str
    prev_snapshot_ts: str
    details: dict = field(default_factory=dict)
    virtual_query: str = ""
    detected_at: str = ""


# ── Diff Computation ───────────────────────────────────────────


def compute_snapshot_diff(
    prev: ClusterSnapshot,
    curr: ClusterSnapshot,
) -> SnapshotDiff:
    """Compute a structured diff between two snapshots."""
    prev_count = len(prev.user_assignments)
    curr_count = len(curr.user_assignments)
    delta = curr_count - prev_count
    pct = (delta / prev_count * 100) if prev_count > 0 else (100.0 if curr_count > 0 else 0.0)

    # Per-segment deltas with percentages
    seg_deltas: dict[int, dict] = {}
    raw_deltas = compare_snapshots(prev, curr)
    for seg, metrics in raw_deltas.items():
        prev_users = prev.segment_stats.get(seg, {}).get("user_count", 0)
        user_delta = metrics.get("user_count_delta", 0)
        seg_deltas[seg] = {
            **metrics,
            "user_count_pct": (user_delta / prev_users * 100) if prev_users > 0 else 0.0,
        }

    # Flow deltas (churned per segment)
    flow_deltas: dict[int, dict] = {}
    for seg in set(list(prev.flow_distribution.keys()) + list(curr.flow_distribution.keys())):
        pf = prev.flow_distribution.get(seg, {})
        cf = curr.flow_distribution.get(seg, {})
        flow_deltas[seg] = {}
        for tag in ["active", "potential", "dormant", "churned"]:
            pv = pf.get(tag, 0)
            cv = cf.get(tag, 0)
            flow_deltas[seg][f"{tag}_delta"] = cv - pv

    # Overall monetary/frequency deltas
    prev_mon = sum(
        s.get("avg_monetary", 0) * s.get("user_count", 0)
        for s in prev.segment_stats.values()
    )
    curr_mon = sum(
        s.get("avg_monetary", 0) * s.get("user_count", 0)
        for s in curr.segment_stats.values()
    )
    mon_pct = ((curr_mon - prev_mon) / prev_mon * 100) if prev_mon > 0 else 0.0

    prev_freq = sum(
        s.get("avg_frequency", 0) * s.get("user_count", 0)
        for s in prev.segment_stats.values()
    )
    curr_freq = sum(
        s.get("avg_frequency", 0) * s.get("user_count", 0)
        for s in curr.segment_stats.values()
    )
    freq_pct = ((curr_freq - prev_freq) / prev_freq * 100) if prev_freq > 0 else 0.0

    diff = SnapshotDiff(
        prev_timestamp=prev.timestamp,
        curr_timestamp=curr.timestamp,
        prev_total_users=prev_count,
        curr_total_users=curr_count,
        total_users_delta=delta,
        total_users_pct_change=round(pct, 2),
        segment_deltas=seg_deltas,
        flow_deltas=flow_deltas,
        avg_monetary_delta_pct=round(mon_pct, 2),
        avg_frequency_delta_pct=round(freq_pct, 2),
    )
    logger.info("snapshot_diff_computed", extra={
        "prev_users": prev_count, "curr_users": curr_count,
        "delta_pct": round(pct, 2),
    })
    return diff


# ── Event Detection ────────────────────────────────────────────


def detect_events(
    diff: SnapshotDiff,
    prev_snapshot: ClusterSnapshot,
    curr_snapshot: ClusterSnapshot,
) -> list[WatcherEvent]:
    """Apply threshold rules to detect business events."""
    from config.settings import get_settings
    settings = get_settings()
    events: list[WatcherEvent] = []
    now = datetime.now(timezone.utc).isoformat()
    prev_ts = diff.prev_timestamp
    curr_ts = diff.curr_timestamp

    # 1. High-value churn (HIGH)
    top_seg = max(curr_snapshot.segment_stats.keys()) if curr_snapshot.segment_stats else None
    if top_seg is not None:
        churned_delta = diff.flow_deltas.get(top_seg, {}).get("churned_delta", 0)
        if churned_delta >= settings.HIGH_VALUE_CHURN_THRESHOLD:
            prev_churned = prev_snapshot.flow_distribution.get(top_seg, {}).get("churned", 0)
            curr_churned = curr_snapshot.flow_distribution.get(top_seg, {}).get("churned", 0)
            pct = (churned_delta / prev_churned * 100) if prev_churned > 0 else 100
            details = {
                "top_segment": top_seg, "churn_count": churned_delta,
                "prev_churned": prev_churned, "curr_churned": curr_churned,
                "churn_pct": round(pct, 1),
            }
            events.append(WatcherEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.HIGH_VALUE_CHURN,
                priority=Priority.HIGH,
                source_snapshot_ts=curr_ts, prev_snapshot_ts=prev_ts,
                details=details,
                virtual_query=build_virtual_query(EventType.HIGH_VALUE_CHURN, details, prev_ts, curr_ts),
                detected_at=now,
            ))

    # 2. Order volume crash (HIGH)
    if diff.total_users_pct_change <= -settings.ORDER_CRASH_THRESHOLD_PCT:
        details = {
            "total_drop_pct": abs(diff.total_users_pct_change),
            "monetary_delta_pct": diff.avg_monetary_delta_pct,
            "prev_total": diff.prev_total_users, "curr_total": diff.curr_total_users,
        }
        events.append(WatcherEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ORDER_VOLUME_CRASH,
            priority=Priority.HIGH,
            source_snapshot_ts=curr_ts, prev_snapshot_ts=prev_ts,
            details=details,
            virtual_query=build_virtual_query(EventType.ORDER_VOLUME_CRASH, details, prev_ts, curr_ts),
            detected_at=now,
        ))

    # 3. Extreme outlier (segment monetary std deviation)
    mon_values = [
        s.get("avg_monetary", 0) for s in curr_snapshot.segment_stats.values()
    ]
    if len(mon_values) >= 2:
        import numpy as np
        mean_mon = np.mean(mon_values)
        std_mon = np.std(mon_values)
        if std_mon > 0:
            for seg, stats in curr_snapshot.segment_stats.items():
                prev_stats = prev_snapshot.segment_stats.get(seg, {})
                old_val = prev_stats.get("avg_monetary", 0)
                new_val = stats.get("avg_monetary", 0)
                z = abs(new_val - mean_mon) / std_mon
                if z > settings.EXTREME_OUTLIER_STD:
                    details = {
                        "seg_id": seg, "metric_name": "avg_monetary",
                        "std_count": round(z, 1),
                        "old_val": round(old_val, 2), "new_val": round(new_val, 2),
                    }
                    events.append(WatcherEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.EXTREME_OUTLIER,
                        priority=Priority.HIGH,
                        source_snapshot_ts=curr_ts, prev_snapshot_ts=prev_ts,
                        details=details,
                        virtual_query=build_virtual_query(EventType.EXTREME_OUTLIER, details, prev_ts, curr_ts),
                        detected_at=now,
                    ))

    # 4. Segment ratio shift (NORMAL)
    for seg, d in diff.segment_deltas.items():
        pct = abs(d.get("user_count_pct", 0))
        if pct >= settings.SEGMENT_RATIO_THRESHOLD_PCT:
            stats = curr_snapshot.segment_stats.get(seg, {})
            flow = curr_snapshot.flow_distribution.get(seg, {})
            details = {
                "seg_id": seg,
                "old_count": stats.get("user_count", 0) - d.get("user_count_delta", 0),
                "new_count": stats.get("user_count", 0),
                "delta_pct": round(pct, 1),
                "avg_recency": round(stats.get("avg_recency", 0), 1),
                "avg_monetary": round(stats.get("avg_monetary", 0), 2),
                "flow_tags": str(flow),
            }
            events.append(WatcherEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.SEGMENT_SHIFT,
                priority=Priority.NORMAL,
                source_snapshot_ts=curr_ts, prev_snapshot_ts=prev_ts,
                details=details,
                virtual_query=build_virtual_query(EventType.SEGMENT_SHIFT, details, prev_ts, curr_ts),
                detected_at=now,
            ))

    # 5. Average order value change (NORMAL)
    if abs(diff.avg_monetary_delta_pct) >= settings.EVENT_THRESHOLD_PCT:
        seg_breakdown = ", ".join(
            f"seg{s}: {d.get('avg_monetary_delta', 0):+.1f}"
            for s, d in diff.segment_deltas.items()
        )
        details = {
            "delta_pct": round(diff.avg_monetary_delta_pct, 1),
            "old_avg": round(
                sum(s.get("avg_monetary", 0) * s.get("user_count", 0)
                    for s in prev_snapshot.segment_stats.values()) / max(prev_snapshot.user_assignments.__len__(), 1), 2
            ),
            "new_avg": round(
                sum(s.get("avg_monetary", 0) * s.get("user_count", 0)
                    for s in curr_snapshot.segment_stats.values()) / max(curr_snapshot.user_assignments.__len__(), 1), 2
            ),
            "segment_breakdown": seg_breakdown,
        }
        events.append(WatcherEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.AVG_ORDER_CHANGE,
            priority=Priority.NORMAL,
            source_snapshot_ts=curr_ts, prev_snapshot_ts=prev_ts,
            details=details,
            virtual_query=build_virtual_query(EventType.AVG_ORDER_CHANGE, details, prev_ts, curr_ts),
            detected_at=now,
        ))

    # 6. New user spike (NORMAL)
    if diff.total_users_pct_change >= settings.EVENT_THRESHOLD_PCT:
        seg_breakdown = ", ".join(
            f"seg{s}: {d.get('user_count_delta', 0):+d}"
            for s, d in diff.segment_deltas.items()
        )
        details = {
            "old_total": diff.prev_total_users,
            "new_total": diff.curr_total_users,
            "delta_pct": round(diff.total_users_pct_change, 1),
            "segment_breakdown": seg_breakdown,
        }
        events.append(WatcherEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.NEW_USER_SPIKE,
            priority=Priority.NORMAL,
            source_snapshot_ts=curr_ts, prev_snapshot_ts=prev_ts,
            details=details,
            virtual_query=build_virtual_query(EventType.NEW_USER_SPIKE, details, prev_ts, curr_ts),
            detected_at=now,
        ))

    if events:
        logger.info("events_detected", extra={
            "count": len(events),
            "types": [e.event_type.value for e in events],
        })
    return events


# ── 流失预警(用户级规则,独立于快照 diff)────────────────────────


def detect_high_value_dormant(
    rfm,
    settings,
    prev_reported: set[int] | None = None,
) -> WatcherEvent | None:
    """最高价值分群中"新进入 30 天未下单"的用户数 ≥ 阈值 → HIGH 事件。

    与 high_value_churn 的语义边界:后者基于快照 diff 的 flow_tag=churned
    增量(已确认流失);前者是 recency > HIGH_VALUE_DORMANT_DAYS 的预警。
    两个事件可同轮共存,互不干扰。

    Args:
        rfm: RFM DataFrame(含 segment / recency / user_id 列,来自分群流水线)。
        settings: 配置(HIGH_VALUE_DORMANT_DAYS / MIN_USERS)。
        prev_reported: 已上报过的用户集合(缺省空集 → 纯函数,便于单测)。
            只有"新进入"的用户数 ≥ MIN_USERS 才触发,同一批用户不重复上报。

    Returns:
        HIGH 优先级的 WatcherEvent;条件不满足返回 None。
    """
    import pandas as pd

    prev_reported = prev_reported or set()
    if rfm is None or getattr(rfm, "empty", True):
        return None
    if "segment" not in rfm.columns or "recency" not in rfm.columns:
        return None

    if rfm["segment"].isna().all():
        return None
    max_seg = int(rfm["segment"].max())
    high = rfm[rfm["segment"] == max_seg]
    if high.empty:
        return None

    dormant_days = int(settings.HIGH_VALUE_DORMANT_DAYS)
    dormant = high[high["recency"] > dormant_days]
    total_dormant = len(dormant)
    new_ids = [
        int(uid) for uid in dormant["user_id"].tolist()
        if int(uid) not in prev_reported
    ]
    if len(new_ids) < int(settings.HIGH_VALUE_DORMANT_MIN_USERS):
        return None

    now = datetime.now(timezone.utc).isoformat()
    details = {
        "top_segment": max_seg,
        "dormant_days": dormant_days,
        "new_dormant_count": len(new_ids),
        "new_dormant_ids": new_ids,
        "sample_ids": new_ids[:10],
        "total_dormant_in_seg": total_dormant,
    }
    event = WatcherEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.HIGH_VALUE_DORMANT,
        priority=Priority.HIGH,
        source_snapshot_ts=now,
        prev_snapshot_ts="",
        details=details,
        virtual_query=build_virtual_query(
            EventType.HIGH_VALUE_DORMANT, details, "", now),
        detected_at=now,
    )
    logger.info("high_value_dormant_detected", extra={
        "top_segment": max_seg, "new_count": len(new_ids),
        "total_dormant": total_dormant, "days": dormant_days,
    })
    return event


# ── Dedup / Noise ───────────────────────────────────────────────


def should_suppress(
    event: WatcherEvent,
    task_manager: "TaskManager",  # noqa: F821
    cooldown_seconds: int,
) -> bool:
    """HIGH events never suppressed; NORMAL events check cooldown."""
    if event.priority == Priority.HIGH:
        return False
    return task_manager.is_event_in_cooldown(
        event.event_type.value, cooldown_seconds,
    )


def merge_low_priority(
    events: list[WatcherEvent],
    merge_window_seconds: int,
) -> list[WatcherEvent]:
    """Merge NORMAL events within the merge window into one."""
    high = [e for e in events if e.priority == Priority.HIGH]
    normal = [e for e in events if e.priority == Priority.NORMAL]

    if len(normal) < 2:
        return high + normal

    normal.sort(key=lambda e: e.detected_at)
    merged: list[WatcherEvent] = list(high)
    current_group = [normal[0]]

    for e in normal[1:]:
        # Compare against the *last* event in the group, not the first
        last_dt = datetime.fromisoformat(current_group[-1].detected_at)
        curr_dt = datetime.fromisoformat(e.detected_at)
        if (curr_dt - last_dt).total_seconds() <= merge_window_seconds:
            current_group.append(e)
        else:
            merged.append(_finalize_group(current_group))
            current_group = [e]
    merged.append(_finalize_group(current_group))

    return merged


def _finalize_group(group: list[WatcherEvent]) -> WatcherEvent:
    if len(group) == 1:
        return group[0]
    now = datetime.now(timezone.utc).isoformat()
    preamble = "检测到以下多个用户分群结构变化，请综合分析：\n\n"
    sub_queries = "\n\n---\n\n".join(
        f"{i+1}. {e.virtual_query}" for i, e in enumerate(group)
    )
    return WatcherEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.MERGED_LOW_PRIORITY,
        priority=Priority.NORMAL,
        source_snapshot_ts=group[0].source_snapshot_ts,
        prev_snapshot_ts=group[0].prev_snapshot_ts,
        details={"merged_event_ids": [e.event_id for e in group]},
        virtual_query=preamble + sub_queries,
        detected_at=now,
    )


# ── Virtual Query Templates ────────────────────────────────────


VIRTUAL_QUERY_TEMPLATES: dict[EventType, str] = {
    EventType.HIGH_VALUE_CHURN: (
        "检测到高价值用户流失：从 {prev_ts} 到 {curr_ts} 的监测窗口内，"
        "segment {top_segment}（最高价值分群）中有 {churn_count} 名用户"
        "转为流失状态（churned），流失率变化 {churn_pct}%。"
        "请分析可能的原因，并给出具体的运营挽回建议（如优惠券策略、触达方式等）。"
    ),
    EventType.ORDER_VOLUME_CRASH: (
        "检测到订单活跃度异常下降：从 {prev_ts} 到 {curr_ts} 的监测窗口内，"
        "平台总活跃用户数下降 {total_drop_pct}%，平均消费金额变化 {monetary_delta_pct}%。"
        "请分析可能导致订单量下降的原因（季节性、竞品活动、用户体验等），"
        "并给出应对策略。"
    ),
    EventType.EXTREME_OUTLIER: (
        "检测到异常数据波动：segment {seg_id} 的 {metric_name} 指标"
        "偏离均值 {std_count} 个标准差（从 {old_val} 变为 {new_val}）。"
        "请判断这是否为数据异常或真实的业务突变，并分析可能的原因。"
    ),
    EventType.SEGMENT_SHIFT: (
        "检测到用户分群结构变化：segment {seg_id} 的用户数"
        "从 {old_count} 变为 {new_count}（变化 {delta_pct}%）。"
        "同期该分群的平均近度为 {avg_recency} 天，平均消费为 {avg_monetary} 元，"
        "流转标签分布：{flow_tags}。请分析可能的原因和后续影响。"
    ),
    EventType.AVG_ORDER_CHANGE: (
        "检测到平均消费金额变化：从 {prev_ts} 到 {curr_ts} 的监测窗口内，"
        "全平台用户平均消费金额变化 {delta_pct}%（从 {old_avg} 变为 {new_avg} 元）。"
        "各分群平均消费变化：{segment_breakdown}。请分析原因和影响。"
    ),
    EventType.NEW_USER_SPIKE: (
        "检测到用户基数快速增长：用户总数从 {old_total} 增长到 {new_total}"
        "（增长 {delta_pct}%）。各分群用户数变化：{segment_breakdown}。"
        "请分析新用户的可能来源（拉新活动、自然增长等），"
        "并评估对现有分群结构的影响。"
    ),
    EventType.HIGH_VALUE_DORMANT: (
        "检测到最高价值分群(segment {top_segment})中新增 {new_dormant_count} 名用户"
        "已 {dormant_days} 天未下单，进入流失预警(用户样本: {sample_ids})。"
        "请分析可能原因(价格、竞品、需求周期等)，并制定召回策略："
        "触达方式、优惠力度、时间窗口。"
    ),
}


def build_virtual_query(
    event_type: EventType,
    details: dict,
    prev_ts: str,
    curr_ts: str,
) -> str:
    """Render a concrete Chinese question from template + details."""
    template = VIRTUAL_QUERY_TEMPLATES.get(event_type, "请分析以下数据变化：{raw_summary}")
    ns = {"prev_ts": prev_ts[:19], "curr_ts": curr_ts[:19], **details}
    try:
        return template.format(**ns)
    except KeyError:
        return f"请分析数据变化：{details}"
