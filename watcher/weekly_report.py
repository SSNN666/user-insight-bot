"""自动周报:每周一生成 Markdown 周报(纯规则拼接,不调 LLM)。

数据源(全部为已持久化数据,零网络):
  一、本周事件清单:watcher_tasks 表(created_at 时间窗口,走 idx_tasks_type_ts 索引);
  二、分群规模与趋势:快照文件(load_snapshots 时间窗口,首末对比);
  三、高价值分群重点:最新快照的 max segment + 流转分布;
  四、运营建议:规则生成(负增长分群排查 / 低转化触达 / 流失预警召回)。

调度:WeeklyReportScheduler(仿 flywheel/scheduler.py 的 stop_event + wait_for 模式)——
  start() 时补做(已过本周触发时刻且本周文件缺失 → 立即生成,错过周一不空白);
  主循环等待下一个周一触发时刻。幂等:同周文件已存在且非 force → 跳过。

时区:WEEKLY_REPORT_TIMEZONE(默认 Asia/Shanghai)决定"本周一"边界与触发时刻;
事件窗口边界转 UTC ISO 与 watcher_tasks.created_at(UTC) 比较。
注意:Windows 无系统 tzdata,Asia/Shanghai 固定按 UTC+8 处理(无夏令时)。
"""

import asyncio
import os
from datetime import datetime, time, timedelta, timezone
from typing import Iterable

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

REPORT_DIR_NAME = "weekly_reports"

_UTC8 = timezone(timedelta(hours=8))


def _tz() -> timezone:
    """周报时区:Asia/Shanghai 等东八区固定 UTC+8;其余尝试 zoneinfo,失败回退 UTC+8。"""
    tz_name = get_settings().WEEKLY_REPORT_TIMEZONE
    if tz_name in ("Asia/Shanghai", "Asia/Chongqing", "PRC", "Etc/GMT-8"):
        return _UTC8
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        return _UTC8


def _iso_week_str(dt: datetime) -> str:
    """ISO 周标识,如 2026-W33。"""
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def _week_bounds(now: datetime, tz: timezone) -> tuple[datetime, datetime, str, str]:
    """(本周一00:00 本地, now 本地, since_utc_iso, until_utc_iso)。

    since/until 为 UTC ISO(与 watcher_tasks.created_at 格式一致,可字符串比较)。
    """
    local = now.astimezone(tz)
    monday = (local - timedelta(days=local.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0)
    since_utc = monday.astimezone(timezone.utc).isoformat()
    until_utc = local.astimezone(timezone.utc).isoformat()
    return monday, local, since_utc, until_utc


def _next_report_time(now: datetime, hour: int, tz: timezone) -> datetime:
    """下一个周报触发时刻:今天周一且未过 hour → 今天 hour;否则下周一 hour:00(本地)。"""
    local = now.astimezone(tz)
    if local.weekday() == 0 and (local.hour, local.minute, local.second) < (hour, 0, 0):
        return local.replace(hour=hour, minute=0, second=0, microsecond=0)
    next_monday = local + timedelta(days=7 - local.weekday())
    return next_monday.replace(hour=hour, minute=0, second=0, microsecond=0)


# ── 报告正文 ────────────────────────────────────────────────────


def _build_content(
    week: str,
    monday_local: datetime,
    now_local: datetime,
    monday_naive: str,
) -> str:
    """纯规则拼接四节 Markdown(快照时间戳为 naive 本地,用 monday_naive 过滤)。"""
    from watcher.task_manager import get_task_manager
    from pipeline.user_segmentation import load_snapshots
    from pipeline.profile import compute_segment_growth
    from pipeline.segment_naming import get_segment_names

    lines = [
        f"# 📊 运营周报 {week}",
        f"**周期**: {monday_local.date()} ~ {now_local.date()} | "
        f"**生成**: {now_local.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # ── 一、本周事件清单 ──
    lines.append("## 一、本周自动监测事件")
    since_utc = monday_local.astimezone(timezone.utc).isoformat()
    until_utc = now_local.astimezone(timezone.utc).isoformat()
    tasks = get_task_manager().list_tasks_by_time(since_utc, until_utc, limit=200)
    if tasks:
        lines.append("| 事件类型 | 优先级 | 状态 | 时间 | 摘要 |")
        lines.append("|---------|--------|------|------|------|")
        for t in tasks:
            summary = (t.virtual_query or "").replace("\n", " ")[:80].replace("|", "\\|")
            ts = (t.created_at or "")[:16].replace("T", " ")
            lines.append(
                f"| {t.event_type} | {t.priority} | {t.status} | {ts} | {summary} |")
    else:
        lines.append("本周无自动监测事件(业务运行平稳)。")
    lines.append("")

    # ── 二、分群规模与趋势(快照首末对比) ──
    lines.append("## 二、分群规模与趋势")
    snaps = load_snapshots(since=monday_naive)
    degraded = False
    if len(snaps) < 2:
        all_snaps = load_snapshots()
        snaps = all_snaps[-2:] if len(all_snaps) >= 2 else all_snaps
        degraded = len(all_snaps) >= 2
    if len(snaps) < 2:
        lines.append("快照不足(需≥2个),本周暂无分群趋势数据。")
    else:
        prev, curr = snaps[-2], snaps[-1]
        growth = compute_segment_growth(prev.segment_stats, curr.segment_stats)
        names = get_segment_names()
        note = " (快照不足,使用最近两期)" if degraded else ""
        lines.append(f"分群变化对比: {prev.timestamp[:16].replace('T', ' ')} → "
                     f"{curr.timestamp[:16].replace('T', ' ')}{note}")
        lines.append("")
        lines.append("| 分群 | 人数 | 人数变化 | 平均消费 | 销售额增长(%) | 转化率变化(%) |")
        lines.append("|------|------|---------|---------|--------------|--------------|")
        for g in growth:
            label = names.get(int(g["segment"]), f"分群{g['segment']}")
            delta = g.get("curr_user_count", g["user_count"]) - g.get("prev_user_count", g["user_count"])
            lines.append(
                f"| {label} | {g['user_count']} | {delta:+d} | "
                f"¥{g['avg_monetary']:.2f} | {g['sales_growth_pct']:+.1f} | "
                f"{g['conversion_change_pct']:+.1f} |")
        lines.append("")
    lines.append("")

    # ── 三、高价值分群重点 ──
    lines.append("## 三、高价值分群重点")
    if snaps:
        last = snaps[-1]
        if last.segment_stats:
            top_seg = max(last.segment_stats.keys())
            st = last.segment_stats[top_seg]
            flow = last.flow_distribution.get(top_seg, {})
            names = get_segment_names()
            label = names.get(int(top_seg), f"分群{top_seg}")
            lines.append(f"- **{label}**(segment {top_seg}): {st.get('user_count', 0)} 人,"
                         f"平均近度 {st.get('avg_recency', 0)} 天,"
                         f"平均频次 {st.get('avg_frequency', 0)},"
                         f"平均消费 ¥{st.get('avg_monetary', 0):.2f}")
            if flow:
                active = flow.get("active", 0)
                total = sum(flow.values()) or 1
                lines.append(
                    f"- 流转分布: active {active} / potential {flow.get('potential', 0)} / "
                    f"dormant {flow.get('dormant', 0)} / churned {flow.get('churned', 0)}"
                    f"(active 占比 {active / total * 100:.0f}%)")
            if len(snaps) >= 2:
                prev_top = snaps[-2].segment_stats.get(top_seg, {})
                delta = st.get("user_count", 0) - prev_top.get("user_count", 0)
                lines.append(f"- 较上周: {delta:+d} 人")
                if delta < 0:
                    lines.append("  > ⚠️ 高价值分群人数下降,建议关注流失风险。")
            lines.append("")
    else:
        lines.append("暂无快照数据。")
    lines.append("")

    # ── 四、运营建议(规则生成) ──
    lines.append("## 四、运营建议")
    suggestions: list[str] = []
    if len(snaps) >= 2:
        prev, curr = snaps[-2], snaps[-1]
        growth = compute_segment_growth(prev.segment_stats, curr.segment_stats)
        names = get_segment_names()

        def _label(g):
            return names.get(int(g["segment"]), f"分群{g['segment']}")

        worst = min(growth, key=lambda g: g["sales_growth_pct"])
        if worst["sales_growth_pct"] < 0:
            suggestions.append(
                f"- **{_label(worst)}** 销售额负增长({worst['sales_growth_pct']:+.1f}%),"
                f"建议排查品类供给/价格竞争力并安排触达。")
        flat = [g for g in growth if g["conversion_change_pct"] < 0]
        if flat:
            suggestions.append(
                f"- {'、'.join(_label(g) for g in flat)} 转化率(频次代理)负增长,"
                f"建议开展复购激励(优惠券/会员权益)。")
        top_flow = snaps[-1].flow_distribution.get(
            max(snaps[-1].segment_stats.keys()), {}) if snaps[-1].segment_stats else {}
        if top_flow.get("dormant", 0) > 0 or top_flow.get("churned", 0) > 0:
            suggestions.append(
                f"- 高价值分群存在 dormant {top_flow.get('dormant', 0)} / "
                f"churned {top_flow.get('churned', 0)} 用户,建议执行召回策略"
                f"(定向优惠 + 触达窗口)。")
    if not suggestions:
        suggestions.append("- 本周无显著异常,维持现有运营节奏。")
    lines.extend(suggestions)
    lines.append("")
    return "\n".join(lines)


# ── 生成入口(幂等) ──────────────────────────────────────────────


def generate_weekly_report(
    now: datetime | None = None, force: bool = False,
) -> dict:
    """生成本周周报。幂等:同周文件已存在且非 force → 复用,不重复生成。

    Returns: {"week": "2026-W33", "path": str, "content": str, "generated": bool}
    """
    now = now or datetime.now(timezone.utc)
    settings = get_settings()
    tz = _tz()
    monday_local, now_local, _, _ = _week_bounds(now, tz)
    week = _iso_week_str(monday_local)
    report_dir = os.path.join(settings.CACHE_DIR, REPORT_DIR_NAME)
    path = os.path.join(report_dir, f"{week}.md")

    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as f:
            return {"week": week, "path": path, "content": f.read(), "generated": False}

    # 快照时间戳为 naive 本地时间(与 watcher_tasks 的 UTC ISO 不同格式),单独过滤
    monday_naive = monday_local.replace(tzinfo=None).isoformat()
    content = _build_content(week, monday_local, now_local, monday_naive)
    os.makedirs(report_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("weekly_report_generated", extra={"week": week, "path": path})
    return {"week": week, "path": path, "content": content, "generated": True}


def list_recent_reports(limit: int = 5) -> list[dict]:
    """按文件名倒序列出最近生成的周报(文件名即 ISO 周标识,字典序=时间序)。"""
    settings = get_settings()
    report_dir = os.path.join(settings.CACHE_DIR, REPORT_DIR_NAME)
    if not os.path.isdir(report_dir) or limit <= 0:
        return []
    names = sorted(
        fn for fn in os.listdir(report_dir) if fn.endswith(".md")
    )[-limit:]
    out = []
    for fn in reversed(names):
        path = os.path.join(report_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue
        out.append({"week": fn[:-3], "path": path, "content": content})
    return out


# ── 调度器(仿 flywheel/scheduler.py)────────────────────────────


class WeeklyReportScheduler:
    """后台定时任务:错过触发时刻自动补做,主循环等待下个周一。"""

    def __init__(self):
        self.stop_event = asyncio.Event()
        self._running = False

    async def start(self) -> None:
        settings = get_settings()
        self._running = True
        self.stop_event.clear()

        # 补做:已过本周触发时刻且本周文件缺失(错过周一 9 点 → 整周不能空白)
        local = datetime.now(timezone.utc).astimezone(_tz())
        passed = local.weekday() > 0 or (
            local.weekday() == 0 and local.hour >= settings.WEEKLY_REPORT_HOUR)
        if passed:
            try:
                await self.generate_once()
            except Exception:
                logger.exception("weekly_report_backfill_failed")

        logger.info("weekly_report_scheduler_started", extra={
            "hour": settings.WEEKLY_REPORT_HOUR, "tz": settings.WEEKLY_REPORT_TIMEZONE,
        })

        while not self.stop_event.is_set():
            next_t = _next_report_time(
                datetime.now(timezone.utc), settings.WEEKLY_REPORT_HOUR, _tz())
            wait_s = max(1.0, (next_t - datetime.now(timezone.utc)).total_seconds())
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=wait_s)
            except asyncio.TimeoutError:
                pass
            if self.stop_event.is_set():
                break
            try:
                await self.generate_once()
            except Exception:
                logger.exception("weekly_report_generate_failed")

    async def generate_once(self, now: datetime | None = None, force: bool = False) -> dict:
        """生成一次(供调度与手动触发共用)。"""
        return generate_weekly_report(now, force)

    def stop(self) -> None:
        self.stop_event.set()
        self._running = False


# ── Singleton ───────────────────────────────────────────────────

_scheduler: WeeklyReportScheduler | None = None


def get_weekly_report_scheduler() -> WeeklyReportScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = WeeklyReportScheduler()
    return _scheduler
