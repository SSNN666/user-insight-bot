"""回答透出图表数据:Skill 结构化结果 → 前端可渲染的轻量 chart spec。

零依赖的图表协议(前端用内联 SVG 渲染,不引第三方图表库):
  {
    "type": "bar" | "line",
    "title": str,
    "categories": [x 轴标签],
    "series": [{"name": str, "data": [num, ...]}],
  }

规则(纯函数,可单测):
  - 分群统计行(segment + 用户数/平均消费)→ bar:各分群人数与平均消费(业务名标注);
  - 快照时序行(timestamp + segment + user_count)→ line:分群人数时间趋势;
  - 漏斗行(step + user_count + conversion_rate)→ bar:浏览/加购/下单 用户数;
  - 环比明细行(user_count_delta 等)→ 不做图(表格更适合);
  - 数据点/分群数超上限自动截断,避免撑爆前端。
"""
from typing import Any

MAX_CATEGORIES = 24
MAX_SERIES = 8


def _label(seg, names: dict) -> str:
    name = names.get(int(seg))
    return f"分群{seg}·{name}" if name else f"分群{seg}"


def _stats_bar_chart(rows: list[dict], names: dict) -> dict | None:
    """[{'segment':0,'用户数':1852,'平均消费':7678.67,...}] → bar 图。"""
    usable = [
        r for r in rows
        if isinstance(r, dict) and "segment" in r and "用户数" in r and "平均消费" in r
    ]
    if not usable:
        return None
    usable = usable[:MAX_CATEGORIES]
    return {
        "type": "bar",
        "title": "各分群人数与平均消费",
        "categories": [_label(r["segment"], names) for r in usable],
        "series": [
            {"name": "用户数", "data": [r["用户数"] for r in usable]},
            {"name": "平均消费(元)", "data": [round(float(r["平均消费"]), 2) for r in usable]},
        ],
    }


def _trend_line_chart(rows: list[dict], names: dict) -> dict | None:
    """[{'timestamp':'2016-02-01 10:00','segment':0,'user_count':60}] → line 图。"""
    usable = [
        r for r in rows
        if isinstance(r, dict) and "timestamp" in r and "segment" in r
        and r.get("user_count") is not None
    ]
    if not usable:
        return None

    timestamps = sorted({str(r["timestamp"]) for r in usable})[:MAX_CATEGORIES]
    segs = sorted({int(r["segment"]) for r in usable})[:MAX_SERIES]

    def _pt(row: dict, ts: str, seg: int) -> float | None:
        if str(row["timestamp"]) == ts and int(row["segment"]) == seg:
            return round(float(row["user_count"]), 1)
        return None

    series = []
    for seg in segs:
        data = []
        for ts in timestamps:
            for r in usable:
                v = _pt(r, ts, seg)
                if v is not None:
                    data.append(v)
                    break
            else:
                data.append(None)   # 该时间点无此分群 → 前端画断点
        series.append({"name": _label(seg, names), "data": data})
    return {
        "type": "line",
        "title": "分群人数时间趋势",
        "categories": timestamps,
        "series": series,
    }


def _funnel_bar_chart(rows: list[dict]) -> dict | None:
    """[{'step':'浏览','user_count':1200,'conversion_rate':1.0},...] → bar 图。"""
    usable = [
        r for r in rows
        if isinstance(r, dict) and "step" in r and r.get("user_count") is not None
        and "conversion_rate" in r
    ]
    if not usable:
        return None
    usable = usable[:MAX_CATEGORIES]
    return {
        "type": "bar",
        "title": "转化漏斗(浏览→加购→下单)",
        "categories": [str(r["step"]) for r in usable],
        "series": [
            {"name": "用户数", "data": [int(r["user_count"]) for r in usable]},
            {"name": "转化率(%)", "data": [round(r["conversion_rate"] * 100, 1) for r in usable]},
        ],
    }


def build_charts_from_skills(skill_results: list[Any], names: dict | None = None) -> list[dict]:
    """从一轮 SkillResult 中提取图表(规则驱动,最多 2 张,避免刷屏)。"""
    names = names or {}
    charts: list[dict] = []
    for sr in skill_results or []:
        data = getattr(sr, "data", None)
        if not isinstance(data, list) or not data:
            continue
        if charts == []:
            funnel = _funnel_bar_chart(data)
            if funnel:
                charts.append(funnel)
                continue
            bar = _stats_bar_chart(data, names)
            if bar:
                charts.append(bar)
                continue
        line = _trend_line_chart(data, names)
        if line and not any(c.get("type") == "line" for c in charts):
            charts.append(line)
    return charts[:2]
