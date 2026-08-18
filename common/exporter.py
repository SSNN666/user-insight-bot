"""Excel 导出服务:管理台"一键拉数"。

数据全部复用现有模块(零新数据逻辑),生成 .xlsx 到 CACHE_DIR/exports/,
由 api/routes.py 以 FileResponse 交付给管理台下载。

数据集:
  segment_stats  分群统计(含业务命名)
  segment_trend  分群人数时间趋势(最近 10 期快照)
  segment_growth 分群环比增长(最近两期快照)
  segment_users  每个分群的用户名单(多 sheet,按分群)
  funnel         转化漏斗(days 参数,默认 7)
  tasks          自主分析任务列表(可选 status 过滤)
  weekly_report  周报(多 sheet:四节内容)
"""

import os
import re
from datetime import datetime, timezone

import pandas as pd

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

DATASETS = {
    "segment_stats", "segment_trend", "segment_growth", "segment_users",
    "funnel", "tasks", "weekly_report",
}


def _safe_sheet_name(name: str) -> str:
    """Excel sheet 名限制:≤31 字符,禁 []:*?/\。"""
    cleaned = re.sub(r'[\\/*?:\[\]]', '_', str(name))
    return cleaned[:31] or "Sheet"


def _export_dir() -> str:
    path = os.path.join(get_settings().CACHE_DIR, "exports")
    os.makedirs(path, exist_ok=True)
    return path


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ── 各数据集取数 ─────────────────────────────────────────────────


def _segment_stats_df() -> pd.DataFrame:
    from skills.user_segment import _load_and_process
    from pipeline.user_segmentation import segment_summary
    from pipeline.segment_naming import get_segment_names

    _, segments, _ = _load_and_process(force_refresh=False)
    df = segment_summary(segments)
    names = get_segment_names(segments)
    if names:
        df["业务名"] = df["segment"].map(lambda s: names.get(int(s), ""))
    return df


def _segment_trend_df() -> pd.DataFrame:
    from pipeline.user_segmentation import load_snapshots

    rows = []
    for s in load_snapshots()[-10:]:
        ts = s.timestamp[:16].replace("T", " ")
        for sid in sorted(s.segment_stats.keys()):
            st = s.segment_stats[sid]
            rows.append({
                "时间": ts,
                "分群": int(sid),
                "用户数": st.get("user_count", 0),
                "平均消费": round(st.get("avg_monetary", 0), 2),
                "平均近度": st.get("avg_recency", 0),
            })
    return pd.DataFrame(rows, columns=["时间", "分群", "用户数", "平均消费", "平均近度"])


def _segment_growth_df() -> pd.DataFrame:
    from pipeline.user_segmentation import load_snapshots
    from pipeline.profile import compute_segment_growth

    snaps = load_snapshots()
    cols = ["分群", "人数", "销售额增长(%)", "转化率变化(%)", "GMV提升(%)", "人均消费"]
    if len(snaps) < 2:
        return pd.DataFrame(columns=cols)
    prev, curr = snaps[-2], snaps[-1]
    growth = compute_segment_growth(prev.segment_stats, curr.segment_stats)
    rows = [{
        "分群": g["segment"],
        "人数": g["user_count"],
        "销售额增长(%)": g["sales_growth_pct"],
        "转化率变化(%)": g["conversion_change_pct"],
        "GMV提升(%)": g["gmv_lift_pct"],
        "人均消费": round(g["avg_monetary"], 2),
    } for g in growth]
    return pd.DataFrame(rows, columns=cols)


def _funnel_df(days: int = 7) -> pd.DataFrame:
    from pipeline.funnel import load_funnel_actions, compute_funnel

    rows = compute_funnel(load_funnel_actions(), days)
    return pd.DataFrame([{
        "步骤": r["step"],
        "用户数": r["user_count"],
        "转化率": round(r["conversion_rate"] * 100, 1),
        "流失率": round((1 - r["conversion_rate"]) * 100, 1),
    } for r in rows], columns=["步骤", "用户数", "转化率", "流失率"])


def _tasks_df(status: str | None = None) -> pd.DataFrame:
    from watcher.task_manager import get_task_manager

    tasks = get_task_manager().list_tasks(status=status, limit=200)
    rows = [{
        "任务ID": t.id,
        "事件类型": t.event_type,
        "优先级": t.priority,
        "状态": t.status,
        "时间": (t.created_at or "")[:16].replace("T", " "),
        "摘要": (t.virtual_query or "")[:100],
    } for t in tasks]
    return pd.DataFrame(rows, columns=["任务ID", "事件类型", "优先级", "状态", "时间", "摘要"])


def _segment_users_sheets() -> dict[str, pd.DataFrame]:
    """每个分群的用户名单(多 sheet,sheet 名 = 分群号·业务名)。

    数据源为分群流水线的 RFM(含 user_id/recency/frequency/monetary/flow_tag)。
    """
    from skills.user_segment import _load_and_process
    from pipeline.segment_naming import get_segment_names

    _, segments, _ = _load_and_process(force_refresh=False)
    if segments is None or segments.empty or "segment" not in segments.columns:
        return {}

    names = get_segment_names(segments)
    sheets: dict[str, pd.DataFrame] = {}
    for sid in sorted(segments["segment"].unique()):
        seg_df = segments[segments["segment"] == int(sid)]
        label = names.get(int(sid), f"分群{sid}")
        df = pd.DataFrame({
            "用户ID": seg_df["user_id"].astype(int),
            "近度(天)": seg_df["recency"].round(0).astype(int),
            "频次": seg_df["frequency"].round(2),
            "消费金额": seg_df["monetary"].round(2),
        })
        if "flow_tag" in seg_df.columns:
            df["流转标签"] = seg_df["flow_tag"].astype(str)
        sheets[_safe_sheet_name(f"分群{int(sid)}·{label}")] = df
    return sheets


def _weekly_report_sheets() -> dict[str, pd.DataFrame]:
    from watcher.weekly_report import generate_weekly_report

    result = generate_weekly_report()
    content = result["content"]
    # 按 "## " 分节,每节文本每行一个 cell
    sheets: dict[str, pd.DataFrame] = {}
    current = "周报"
    lines_acc: list[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            if lines_acc:
                sheets[current] = pd.DataFrame({"内容": lines_acc})
            current = line[3:].strip()
            lines_acc = []
        elif line.strip():
            lines_acc.append(line)
    if lines_acc:
        sheets[current] = pd.DataFrame({"内容": lines_acc})
    return sheets or {"周报": pd.DataFrame({"内容": [content]})}


# ── 统一入口 ─────────────────────────────────────────────────────


def _dataset_frame(dataset: str, kwargs: dict) -> pd.DataFrame:
    if dataset == "segment_stats":
        return _segment_stats_df()
    if dataset == "segment_trend":
        return _segment_trend_df()
    if dataset == "segment_growth":
        return _segment_growth_df()
    if dataset == "funnel":
        return _funnel_df(days=int(kwargs.get("days", 7)))
    if dataset == "tasks":
        return _tasks_df(status=kwargs.get("status"))
    raise ValueError(f"未知数据集: {dataset}(可用: {sorted(DATASETS)})")


def export_dataset(dataset: str, **kwargs) -> tuple[str, str]:
    """生成指定数据集的 Excel 文件。

    Args:
        dataset: DATASETS 之一。
        kwargs: days(漏斗)/status(任务)等数据集参数。

    Returns:
        (filename, filepath)。非法 dataset 抛 ValueError。
    """
    if dataset not in DATASETS:
        raise ValueError(f"未知数据集: {dataset}(可用: {sorted(DATASETS)})")

    filename = f"{dataset}_{_ts()}.xlsx"
    filepath = os.path.join(_export_dir(), filename)

    if dataset in ("weekly_report", "segment_users"):
        sheets = (_weekly_report_sheets() if dataset == "weekly_report"
                  else _segment_users_sheets())
        if not sheets:
            sheets = {"数据": pd.DataFrame(columns=["提示"])}
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=_safe_sheet_name(name), index=False)
    else:
        df = _dataset_frame(dataset, kwargs)
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="数据", index=False)

    logger.info("excel_exported", extra={"dataset": dataset, "path": filepath})
    return filename, filepath
