"""Gradio frontend — three-panel layout, all calls via httpx to FastAPI.

Phase 6: complete decoupling from business logic.
Tab 1: AI Chat + feedback
Tab 2: Debug Console (CRUD + mock reset + event trigger)
Tab 3: Task Monitor + charts
"""

import os
import warnings

import gradio as gr
import httpx
import matplotlib
from log.logger import get_logger

logger = get_logger("gradio")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# ── 中文字体配置（局部 rcParams，不污染全局 matplotlib 状态）──
_CHINESE_FONT_RC = {
    "font.sans-serif": [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans",
    ],
    "axes.unicode_minus": False,
}

# Apply to current process for Gradio (this is a standalone app, not a library)
plt.rcParams.update(_CHINESE_FONT_RC)

# ── Config ─────────────────────────────────────────────────────
from config.settings import get_settings as _gs
API = f"http://{_gs().API_HOST}:{_gs().API_PORT}"

# Shared HTTP client with connection pool — reused across all Gradio tabs.
# Closed on process exit via atexit.  Never instantiate another httpx.Client
# in chart/refresh helpers; use the module-level helpers below.
_client = httpx.Client(
    timeout=120.0,
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=20,
        keepalive_expiry=30.0,
    ),
)

import atexit
atexit.register(_client.close)

def _post(path: str, json: dict) -> dict:
    return _client.post(f"{API}{path}", json=json).json()

def _get(path: str, params: dict = None) -> dict:
    return _client.get(f"{API}{path}", params=params).json()

def _put(path: str, json: dict) -> dict:
    return _client.put(f"{API}{path}", json=json).json()

def _patch(path: str, json: dict) -> dict:
    return _client.patch(f"{API}{path}", json=json).json()

def _delete(path: str) -> dict:
    return _client.delete(f"{API}{path}").json()

def _load_failed(e: Exception) -> tuple:
    """图表加载失败的统一描述:连接类错误给启动提示,其余透出真实原因。"""
    if isinstance(e, httpx.ConnectError):
        return None, "_加载失败：FastAPI(:8000)未运行——请先启动 run_api.py,或重新双击 start.bat_"
    return None, f"_加载失败：{e}_"

def _seg_names() -> dict:
    """分群业务命名(LLM 生成 + 启发式兜底,带缓存),图表标签共用。"""
    try:
        from pipeline.segment_naming import get_segment_names
        return get_segment_names()
    except Exception:
        return {}

def _seg_label(seg, names: dict) -> str:
    name = names.get(int(seg))
    return f"分群 {seg} · {name}" if name else f"分群 {seg}"

def _freshness_line() -> str:
    """数据新鲜度标注:数据时间 / 数据源 / 分群命名——运营敢用数据的前提。"""
    from config.settings import get_settings
    settings = get_settings()
    parts = []
    try:
        from pipeline.user_segmentation import load_snapshots
        snaps = load_snapshots()
        if snaps:
            parts.append(f"🕐 数据时间: {snaps[-1].timestamp[:16].replace('T', ' ')}")
    except Exception:
        pass
    if settings.DATA_SOURCE == "tianchi":
        cap = f"(行为采样 {settings.TIANCHI_MAX_ACTIONS:,} 行)" if settings.TIANCHI_MAX_ACTIONS else "(全量)"
        parts.append(f"🗄️ 数据源: JData 真实数据 {cap}")
    else:
        parts.append(f"🗄️ 数据源: {'MySQL→缓存→mock 降级链' if settings.DATA_SOURCE == 'auto' else settings.DATA_SOURCE}")
    names = _seg_names()
    if names:
        parts.append("🏷️ " + " / ".join(f"分群{k}·{v}" for k, v in sorted(names.items())))
    return "  \n".join(parts) + "  \n"

# ── Tab 1: Chat ────────────────────────────────────────────────

# Store last exchange for feedback
_last_qa: dict = {"question": "", "reply": ""}

def _chart_spec_to_fig(spec: dict | None):
    """chart 协议(agent/chart_extract.py)→ matplotlib 图,Gradio 对话内展示。"""
    if not spec or not spec.get("categories"):
        return None
    cats = spec.get("categories", [])
    series = spec.get("series", [])
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8B5CF6",
              "#E5C641", "#13C2C2", "#fa8c16", "#722ed1"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if spec.get("type") == "bar":
        x = np.arange(len(cats))
        w = 0.7 / max(len(series), 1)
        for i, s in enumerate(series):
            data = [v if v is not None else 0 for v in s["data"]]
            ax.bar(x + (i - (len(series) - 1) / 2) * w, data, w,
                   label=s["name"], color=colors[i % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=8)
    else:
        for i, s in enumerate(series):
            xs = [j for j, v in enumerate(s["data"]) if v is not None]
            ys = [v for v in s["data"] if v is not None]
            ax.plot(xs, ys, "o-", linewidth=2, markersize=4,
                    label=s["name"], color=colors[i % len(colors)])
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=7)
    ax.set_title(spec.get("title", ""), fontweight="bold", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

_ANALYSIS_KEYWORDS = ("分群", "统计", "消费", "人数", "趋势", "分析", "群", "用户", "价值", "对比", "图")

def _fallback_charts() -> list[dict]:
    """Agent 未调工具(如从会话记忆直接回答)时,直接拉端点建图。

    确定性兜底:图表始终与最新数据同步,不依赖 LLM 是否调用 Skill。
    统计端点 → 柱状图;快照端点 → 趋势折线图(≥2 个快照时)。
    """
    charts: list[dict] = []
    try:
        rows = _get("/stats/rfm", params={"force": False})
        if rows:
            names = _seg_names()
            charts.append({
                "type": "bar",
                "title": "各分群人数与平均消费",
                "categories": [_seg_label(r["segment"], names) for r in rows],
                "series": [
                    {"name": "用户数", "data": [r["user_count"] for r in rows]},
                    {"name": "平均消费(元)", "data": [r["avg_monetary"] for r in rows]},
                ],
            })
    except Exception:
        pass
    try:
        trend = _get("/stats/segment-trend", params={"limit": 20})
        if trend and len({t["timestamp"] for t in trend}) >= 2:
            names = _seg_names()
            timestamps = sorted({t["timestamp"] for t in trend})
            segs = sorted({t["segment"] for t in trend})
            series = []
            for seg in segs:
                pts = {t["timestamp"]: t.get("user_count") for t in trend
                       if t["segment"] == seg}
                series.append({
                    "name": _seg_label(seg, names),
                    "data": [pts.get(ts) for ts in timestamps],
                })
            charts.append({
                "type": "line",
                "title": "分群人数时间趋势",
                "categories": timestamps,
                "series": series,
            })
    except Exception:
        pass
    return charts

def chat_fn(message, history):
    global _last_qa
    bar_fig = line_fig = None
    try:
        # 管理台 = 运营分析台:强制分析模式,不做购物路由
        data = _post("/ask", {"question": message, "session_id": "gradio-chat",
                              "mode": "analysis"})
        if isinstance(data, dict) and "error" in data:
            # 中间件拒绝(限流/重复问题等):把原因如实显示给用户
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            reply = f"⚠️ {msg or '请求被拒绝'}"
        else:
            reply = data.get("reply", "Error: no reply")
            charts = data.get("charts") or []
            # 分析类问题保证双槽有图:Agent 没调工具 → 端点全兜底;
            # Agent 只产出一张(如仅柱状图)→ 端点补齐缺失的类型
            if any(k in message for k in _ANALYSIS_KEYWORDS):
                fb = _fallback_charts()
                have = {c.get("type") for c in charts}
                for c in fb:
                    if c.get("type") not in have:
                        charts.append(c)
            # 按类型路由:bar → 统计图槽,line → 趋势图槽(与标签语义一致)
            bar_fig = next((_chart_spec_to_fig(c) for c in charts if c.get("type") == "bar"), None)
            line_fig = next((_chart_spec_to_fig(c) for c in charts if c.get("type") == "line"), None)
    except Exception as e:
        reply = f"请求失败: {e}"
    _last_qa = {"question": message, "reply": reply}
    return (reply, bar_fig, line_fig)

def feedback_up():
    if _last_qa["question"]:
        _post("/feedback", {**_last_qa, "rating": "up", "session_id": "gradio-chat"})
        return "👍 已记录"
    return "无内容可反馈"

def feedback_down():
    if _last_qa["question"]:
        _post("/feedback", {**_last_qa, "rating": "down", "session_id": "gradio-chat"})
        return "👎 已记录"
    return "无内容可反馈"

# ── Tab 2: Debug Console ───────────────────────────────────────

def debug_add_product(name, category, price):
    try:
        r = _post("/debug/products", {"product_name": name, "category": category, "price": float(price)})
        return f"✅ 商品已创建 (ID: {r.get('product_id')})"
    except Exception as e:
        return f"❌ {e}"

def debug_delete_product(pid):
    try:
        _delete(f"/debug/products/{int(pid)}")
        return f"✅ 商品 {pid} 已下架"
    except Exception as e:
        return f"❌ {e}"

def debug_add_order(uid, pid, qty, amount):
    try:
        r = _post("/debug/orders", {"user_id": int(uid), "product_id": int(pid), "quantity": int(qty), "total_amount": float(amount)})
        return f"✅ 订单已录入 (ID: {r.get('order_id')})"
    except Exception as e:
        return f"❌ {e}"

def debug_update_user(uid, city, age):
    updates = {}
    if city: updates["city"] = city
    if age: updates["age"] = int(age)
    try:
        r = _put(f"/debug/users/{int(uid)}", json=updates)
        return f"✅ 用户 {uid} 已更新: {r.get('updates')}"
    except Exception as e:
        return f"❌ {e}"

def debug_reset_mock():
    try:
        r = _post("/debug/reset-mock", {})
        return f"✅ 数据已重置 — {r.get('users')} 用户, {r.get('segments')} 分群"
    except Exception as e:
        return f"❌ {e}"

def debug_list_products_df():
    """返回当前共享存储中的商品列表"""
    try:
        import pandas as pd
        data = _get("/debug/products")
        if data:
            df = pd.DataFrame(data)
            cols = ["product_id", "product_name", "category", "price"]
            return df[[c for c in cols if c in df.columns]]
        return pd.DataFrame({"提示": ["暂无商品"]})
    except Exception as e:
        return pd.DataFrame({"错误": [str(e)]})


def debug_list_orders_df():
    """返回当前共享存储中的订单列表"""
    try:
        import pandas as pd
        data = _get("/api/orders", params={"page_size": 100})
        orders = data.get("orders", [])
        if orders:
            rows = []
            for o in orders:
                product_names = ", ".join(
                    i.get("product_name", str(i.get("product_id", "?")))
                    for i in (o.get("items", []) if isinstance(o.get("items"), list) else [])
                ) or o.get("product_name", "—")
                rows.append({
                    "order_id": o.get("order_id"),
                    "user_id": o.get("user_id"),
                    "product_name": product_names,
                    "total_amount": o.get("total_amount"),
                    "status": o.get("status"),
                })
            return pd.DataFrame(rows)
        return pd.DataFrame({"提示": ["暂无订单"]})
    except Exception as e:
        return pd.DataFrame({"错误": [str(e)]})


def debug_list_users_df():
    """返回当前共享存储中的用户列表"""
    try:
        import pandas as pd
        data = _get("/debug/users")
        if data:
            df = pd.DataFrame(data)
            cols = ["user_id", "age", "city"]
            return df[[c for c in cols if c in df.columns]]
        return pd.DataFrame({"提示": ["暂无用户"]})
    except Exception as e:
        return pd.DataFrame({"错误": [str(e)]})


def debug_trigger_event():
    try:
        r = _post("/debug/trigger-event", {})
        count = r.get("events_detected", 0)
        return f"✅ 检测完成 — 触发 {count} 个事件" if count else "✅ 检测完成 — 无新事件"
    except Exception as e:
        return f"❌ {e}"

def debug_list_products():
    try:
        import pandas as pd
        data = _get("/debug/products")
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ── Tab 3: Monitor + Charts ────────────────────────────────────

def monitor_list_tasks():
    try:
        import pandas as pd
        data = _get("/tasks/")
        tasks = data.get("tasks", [])
        if tasks:
            df = pd.DataFrame(tasks)
            cols = ["id", "event_type", "priority", "status", "tool_rounds", "created_at"]
            df = df[[c for c in cols if c in df.columns]]
            # 翻译状态列
            if "status" in df.columns:
                status_map = {
                    "pending": "待处理", "running": "运行中", "completed": "已完成",
                    "failed": "失败", "timeout": "超时", "ignored": "已忽略",
                }
                df["status"] = df["status"].map(lambda s: status_map.get(s, str(s)))
            # 翻译事件类型
            if "event_type" in df.columns:
                event_map = {
                    "high_value_churn": "高价值流失", "order_volume_crash": "订单暴跌",
                    "extreme_outlier": "异常波动", "segment_shift": "分群变化",
                    "avg_order_change": "均价变化", "new_user_spike": "新用户激增",
                    "merged_low_priority": "合并事件",
                }
                df["event_type"] = df["event_type"].map(lambda e: event_map.get(e, str(e)))
            # 翻译优先级
            if "priority" in df.columns:
                priority_map = {"high": "🔴 高", "normal": "🟡 普通"}
                df["priority"] = df["priority"].map(lambda p: priority_map.get(p, str(p)))
            return df
        return pd.DataFrame({"info": ["无任务"]})
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def export_excel_file(dataset: str, days: int | None = None) -> str | None:
    """导出数据集为 xlsx:经 API 下载到本地 → 返回文件路径(gr.File 展示)。

    失败返回以 _ 开头的错误文案(供 gr.Markdown 展示)。
    """
    try:
        params = {"dataset": dataset}
        if days is not None:
            params["days"] = days
        resp = _client.get(f"{API}/api/export/excel", params=params, timeout=60)
        if resp.status_code != 200:
            return f"_导出失败(HTTP {resp.status_code}): {resp.text[:150]}_"
        out_dir = os.path.join(_gs().CACHE_DIR, "exports")
        os.makedirs(out_dir, exist_ok=True)
        cd = resp.headers.get("content-disposition", "")
        filename = cd.split("filename=")[-1].strip('"') or f"{dataset}.xlsx"
        path = os.path.join(out_dir, filename)
        with open(path, "wb") as f:
            f.write(resp.content)
        return path
    except Exception as e:
        return f"_导出失败: {e}_"

def monitor_retry_task(task_id):
    try:
        _patch(f"/tasks/{int(task_id)}/retry", json={})
        return f"✅ 任务 {task_id} 已重新加入队列"
    except Exception as e:
        return f"❌ {e}"

def monitor_ignore_task(task_id):
    try:
        _patch(f"/tasks/{int(task_id)}/ignore", json={})
        return f"✅ 任务 {task_id} 已忽略"
    except Exception as e:
        return f"❌ {e}"

def chart_rfm():
    try:
        data = _get("/stats/rfm", params={"force": True})
        if not data:
            return None, "_暂无数据_"
        segs = [d["segment"] for d in data]
        recency = [d["avg_recency"] for d in data]
        freq = [d["avg_frequency"] for d in data]
        mon = [d["avg_monetary"] for d in data]
        names = _seg_names()

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(segs))
        w = 0.25
        ax.bar(x - w, recency, w, label="平均近度 (天)", color="#4C72B0")
        ax.bar(x, freq, w, label="平均频次", color="#55A868")
        ax.bar(x + w, mon, w, label="平均消费 (¥)", color="#C44E52")
        ax.set_xticks(x)
        ax.set_xticklabels([_seg_label(s, names) for s in segs])
        ax.legend(fontsize=8)
        ax.set_title("各分群RFM对比", fontweight="bold")
        fig.tight_layout()

        # Build explanation
        lines = [
            _freshness_line(),
            "**📖 图表说明：RFM 分群对比**",
            "",
            "横向对比每个用户分群在三个核心维度上的平均值：",
            "- **平均近度（天）**：用户最近一次消费距今多少天，**越低越好**，说明用户越活跃；",
            "- **平均频次**：用户平均下单次数，**越高越好**，代表用户粘性强；",
            f"- **平均消费（¥）**：用户平均消费金额，**越高越好**，衡量用户价值。",
            "",
        ]
        # Per-segment summary
        for d in sorted(data, key=lambda x: x["avg_monetary"], reverse=True):
            lines.append(
                f"- {_seg_label(d['segment'], names)}：近度 {d['avg_recency']} 天 · "
                f"频次 {d['avg_frequency']} 次 · 消费 ¥{d['avg_monetary']} · "
                f"共 {d['user_count']} 人"
            )
        return fig, "\n".join(lines)
    except Exception as e:
        return _load_failed(e)

def chart_segment_pie():
    try:
        data = _get("/stats/segment-ratio", params={"force": True})
        if not data:
            return None, "_暂无数据_"
        names = _seg_names()
        labels = [_seg_label(d["segment"], names) for d in data]
        sizes = [d["ratio"] for d in data]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#4C72B0", "#55A868", "#C44E52", "#8B5CF6"][:len(sizes)])
        ax.set_title("分群占比", fontweight="bold")

        lines = [
            _freshness_line(),
            "**📖 图表说明：分群占比**",
            "",
            "展示各分群的用户数量占总体的百分比：",
        ]
        for d in data:
            lines.append(f"- {_seg_label(d['segment'], names)}：{d['count']} 人（占比 {d['ratio']}%）")
        lines.append("")
        lines.append("> 💡 理想情况下各分群占比应相对均衡。若某个分群占比过高或过低，建议关注是否需要调整运营策略。")
        return fig, "\n".join(lines)
    except Exception as e:
        return _load_failed(e)

def chart_radar():
    try:
        data = _get("/stats/rfm", params={"force": True})
        if not data:
            return None, "_暂无数据_"
        categories = ["近度", "频次", "消费"]
        N = len(categories)

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Normalize
        from collections import defaultdict
        norm = defaultdict(list)
        for d in data:
            norm["recency"].append(d["avg_recency"])
            norm["frequency"].append(d["avg_frequency"])
            norm["monetary"].append(d["avg_monetary"])
        r_max, f_max, m_max = max(norm["recency"]) or 1, max(norm["frequency"]) or 1, max(norm["monetary"]) or 1

        colors = ["#4C72B0", "#55A868", "#C44E52", "#8B5CF6"]
        names = _seg_names()
        for d in data:
            vals = [
                d["avg_recency"] / r_max,
                d["avg_frequency"] / f_max,
                d["avg_monetary"] / m_max,
            ]
            vals += vals[:1]
            ax.fill(angles, vals, alpha=0.1, color=colors[d["segment"] % len(colors)])
            ax.plot(angles, vals, "o-", linewidth=2, label=_seg_label(d["segment"], names), color=colors[d["segment"] % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        ax.set_title("分群雷达图", fontweight="bold")
        fig.tight_layout()

        lines = [
            _freshness_line(),
            "**📖 图表说明：分群雷达图**",
            "",
            "用雷达图的形式对比各分群在近度、频次、消费三个维度上的综合表现：",
            "- 每个分群对应一个多边形，面积越大代表整体表现越强；",
            "- **近度** 已经过归一化处理（值越大 = 越活跃 = 越好）；",
            "- 可通过多边形形状快速识别各分群的强项和弱项。",
            "",
            "> 💡 理想的高价值分群应在三个维度上均衡发展，呈三角形扩张趋势。",
        ]
        return fig, "\n".join(lines)
    except Exception as e:
        return _load_failed(e)

def chart_trend():
    try:
        data = _get("/stats/segment-trend", params={"limit": 20})
        if not data:
            return None, "_暂无历史快照数据(等 Watcher 轮询几轮后刷新)_"
        segs = sorted({d["segment"] for d in data})
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8B5CF6"]
        names = _seg_names()
        for seg in segs:
            pts = [d for d in data if d["segment"] == seg]
            x = [d["timestamp"] for d in pts]
            y = [d["user_count"] for d in pts]
            ax.plot(x, y, "o-", linewidth=2, markersize=4,
                    label=_seg_label(seg, names), color=colors[seg % len(colors)])
        ax.set_title("分群人数时间趋势(历史快照)", fontweight="bold")
        ax.set_ylabel("用户数")
        ax.legend(fontsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        fig.tight_layout()

        lines = [
            _freshness_line(),
            "**📖 图表说明:分群人数时间趋势**",
            "",
            "Watcher 引擎每轮轮询都会保存一份分群快照,本图把快照历史画成时间序列:",
            "- **上升/下降拐点** = 事件检测的依据(如分群迁移、新用户激增);",
            "- 数据完全来自磁盘快照,重启不丢失;",
            "- 演示时可以先去「调试操作台」录入几笔订单,再手动触发事件检测,回到本页刷新即可看到新快照点。",
        ]
        return fig, "\n".join(lines)
    except Exception as e:
        return _load_failed(e)

def chart_flow():
    try:
        data = _get("/stats/flow", params={"force": True})
        if not data:
            return None, "_暂无数据_"
        segs = [d["segment"] for d in data]
        tags = ["active", "potential", "dormant", "churned"]
        tag_labels = {"active": "活跃", "potential": "潜力", "dormant": "沉默", "churned": "流失"}
        colors = {"active": "#55A868", "potential": "#4C72B0", "dormant": "#E5C641", "churned": "#C44E52"}

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(segs))
        w = 0.2
        for i, tag in enumerate(tags):
            vals = [d.get(tag, 0) for d in data]
            ax.bar(x + i * w, vals, w, label=tag_labels[tag], color=colors.get(tag, "#888"))

        names = _seg_names()
        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels([_seg_label(s, names) for s in segs])
        ax.legend(fontsize=8)
        ax.set_title("各分群流转分布", fontweight="bold")
        fig.tight_layout()

        lines = [
            _freshness_line(),
            "**📖 图表说明：各分群流转分布**",
            "",
            "展示每个分群内用户的流转状态构成。流转标签含义：",
            "- **活跃**：近期有消费，处于活跃状态；",
            "- **潜力**：有消费记录但近期未下单，存在唤醒潜力；",
            "- **沉默**：长时间未消费，存在流失风险；",
            "- **流失**：已基本不再消费，需要重点关注。",
            "",
            "> 💡 关注各分群的流失/沉默用户占比。高价值分群若流失比例较高，建议优先执行召回策略。",
        ]
        return fig, "\n".join(lines)
    except Exception as e:
        return _load_failed(e)

# ── Build UI ───────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="电商用户智能画像平台") as demo:
        gr.Markdown("# 电商用户智能画像与AI分析平台")
        gr.Markdown("[🛒 电商商城](http://localhost:5173)  |  [📘 API文档](http://localhost:8000/docs)")

        # ════════════════════════════════════════════════════════
        # Tab 1: Chat
        # ════════════════════════════════════════════════════════
        with gr.Tab("💬 AI 对话"):
            gr.Markdown("向 AI 助手询问用户分群信息，所有回答附带反馈按钮。")
            chat = gr.Chatbot(height=380, label="对话")
            with gr.Row():
                chat_input = gr.Textbox(
                    label="问题", scale=4,
                    placeholder="如：各分群的人数是多少？/ 分群人数趋势怎么样？",
                )
                chat_send = gr.Button("发送", variant="primary", scale=1)
                chat_clear = gr.Button("清空对话", size="sm", scale=1)
            with gr.Row():
                chat_plot1 = gr.Plot(label="📊 分群统计图(分析类问题自动出图)")
                chat_plot2 = gr.Plot(label="📈 分群趋势图(问'趋势'类问题出图)")

            def chat_submit(message, history):
                if not message or not message.strip():
                    return history, "", None, None
                reply, bar_fig, line_fig = chat_fn(message, history)
                # Gradio 6 Chatbot 只接受 {"role","content"} 字典格式(旧 tuple 格式会报错)
                history = (history or []) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply},
                ]
                return history, "", bar_fig, line_fig

            chat_send.click(chat_submit,
                            [chat_input, chat],
                            [chat, chat_input, chat_plot1, chat_plot2])
            chat_input.submit(chat_submit,
                              [chat_input, chat],
                              [chat, chat_input, chat_plot1, chat_plot2])
            chat_clear.click(lambda: [], outputs=chat)
            with gr.Row():
                btn_up = gr.Button("👍", size="sm", scale=0)
                btn_down = gr.Button("👎", size="sm", scale=0)
                fb_msg = gr.Textbox(label="反馈状态", interactive=False, scale=3)
            btn_up.click(feedback_up, outputs=fb_msg)
            btn_down.click(feedback_down, outputs=fb_msg)

            # ── 反馈记录 ──
            gr.Markdown("### 📋 反馈记录")
            gr.Markdown("> 点击 👍/👎 后，点「刷新」查看记录是否生效。")
            fb_refresh_btn = gr.Button("🔄 刷新反馈记录", size="sm")
            fb_stats_display = gr.Markdown("_点击刷新查看统计_")
            fb_table = gr.Dataframe(
                label="最近反馈", interactive=False,
                headers=["ID", "问题", "评分", "时间"],
            )

            def refresh_feedback():
                try:
                    stats = _get("/feedback/stats")
                    history = _get("/feedback/history", params={"limit": 20})
                    stats_text = (
                        f"👍 {stats['up']} 个好评  |  👎 {stats['down']} 个差评  |  "
                        f"📊 共 {stats['total']} 条反馈"
                    )
                    rows = []
                    for e in history.get("entries", []):
                        rating_text = "👍" if e["rating"] == "up" else "👎"
                        rows.append({
                            "ID": e["id"],
                            "问题": e["question"][:60],
                            "评分": rating_text,
                            "时间": e["created_at"][:19],
                        })
                    import pandas as pd
                    df = pd.DataFrame(rows) if rows else pd.DataFrame({"提示": ["暂无反馈"]})
                    return stats_text, df
                except Exception as e:
                    return f"_加载失败: {e}_", pd.DataFrame({"错误": [str(e)]})

            fb_refresh_btn.click(refresh_feedback, outputs=[fb_stats_display, fb_table])

            # ── 数字飞轮 ──
            gr.Markdown("### 🔁 数字飞轮")
            gr.Markdown("> 反馈记录 → 自动评分 → 向量入库 → 检索反哺 Agent。点「触发飞轮」立即执行一个完整周期。")
            with gr.Row():
                fw_trigger_btn = gr.Button("⚡ 触发飞轮（采集→评分→入库）", variant="primary", size="sm")
                fw_refresh_btn = gr.Button("🔄 刷新飞轮统计", size="sm")
            fw_stats_display = gr.Markdown("_点击刷新查看飞轮状态_")

            def trigger_flywheel():
                try:
                    r = _post("/flywheel/trigger", json={})
                    return f"✅ 飞轮执行完成 — 本次入库 {r.get('ingested', 0)} 条样本"
                except Exception as e:
                    return f"❌ 飞轮执行失败: {e}"

            def refresh_flywheel_stats():
                try:
                    stats = _get("/flywheel/stats")
                    vs_count = stats.get("vectors_indexed", 0)
                    pos = stats.get("total_positives", 0)
                    neg = stats.get("total_negatives", 0)
                    by_source = stats.get("samples", {})
                    source_lines = "\n".join(
                        f"- {k}: **{v}** 条" for k, v in sorted(by_source.items())
                    ) if by_source else "- _暂无样本_"
                    return (
                        f"**样本库**: {pos + neg} 条（👍 {pos} / 👎 {neg}）  |  "
                        f"**向量索引**: {vs_count} 条\n\n"
                        f"**来源分布**:\n{source_lines}\n\n"
                        f"> 💡 向量索引的样本会被 Agent 检索并注入系统提示，"
                        f"实现「越用越好」。点赞的优质回答会被优先检索。"
                    )
                except Exception as e:
                    return f"_加载失败: {e}_"

            fw_trigger_btn.click(trigger_flywheel, outputs=fw_stats_display)
            fw_refresh_btn.click(refresh_flywheel_stats, outputs=fw_stats_display)

        # ════════════════════════════════════════════════════════
        # Tab 2: Debug Console
        # ════════════════════════════════════════════════════════
        with gr.Tab("🛠️ 调试操作台"):
            gr.Markdown("手动操作模拟商城数据，测试自主 Agent 响应。")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 商品管理")
                    prod_name = gr.Textbox(label="商品名称")
                    prod_cat = gr.Dropdown(["电子产品", "教育用品", "时尚服饰", "家具家居", "生活用品"], label="品类", value="电子产品")
                    prod_price = gr.Number(label="价格 (¥)", value=99.0)
                    with gr.Row():
                        add_btn = gr.Button("新增商品", variant="primary")
                        del_pid = gr.Number(label="商品ID", value=1, precision=0)
                        del_btn = gr.Button("下架商品", variant="stop")
                    prod_result = gr.Textbox(label="操作结果", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("### 订单 & 用户")
                    ord_uid = gr.Number(label="用户ID", value=1, precision=0)
                    ord_pid = gr.Number(label="商品ID", value=1, precision=0)
                    ord_qty = gr.Number(label="数量", value=1, precision=0)
                    ord_amt = gr.Number(label="金额 (¥)", value=99.0)
                    ord_btn = gr.Button("录入订单", variant="primary")
                    ord_result = gr.Textbox(label="操作结果", interactive=False)

                    gr.Markdown("---")
                    usr_uid = gr.Number(label="用户ID", value=1, precision=0)
                    usr_city = gr.Textbox(label="城市")
                    usr_age = gr.Number(label="年龄", value=25, precision=0)
                    usr_btn = gr.Button("修改用户")
                    usr_result = gr.Textbox(label="操作结果", interactive=False)

            with gr.Row():
                reset_btn = gr.Button("🔄 一键重置模拟数据集", variant="secondary")
                trigger_btn = gr.Button("⚡ 手动触发事件检测", variant="secondary")
                refresh_data_btn = gr.Button("🔍 刷新数据预览", variant="secondary")
            debug_result = gr.Textbox(label="系统操作结果", interactive=False)

            # ── 数据预览表格 ──
            gr.Markdown("### 📋 当前数据预览")
            gr.Markdown("> 下方显示的是共享存储中的实时数据。调试台和电商商城共用同一份数据。")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**商品列表**")
                    product_table = gr.Dataframe(label="商品", interactive=False, headers=["ID", "名称", "品类", "价格"], wrap=True)
                with gr.Column(scale=1):
                    gr.Markdown("**订单列表**")
                    order_table = gr.Dataframe(label="订单", interactive=False, headers=["订单号", "用户ID", "商品", "金额", "状态"], wrap=True)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**用户列表**")
                    user_table = gr.Dataframe(label="用户", interactive=False, headers=["用户ID", "年龄", "城市"], wrap=True)

            reset_btn.click(debug_reset_mock, outputs=debug_result)
            trigger_btn.click(debug_trigger_event, outputs=debug_result)
            add_btn.click(debug_add_product, [prod_name, prod_cat, prod_price], prod_result)
            del_btn.click(debug_delete_product, del_pid, prod_result)
            ord_btn.click(debug_add_order, [ord_uid, ord_pid, ord_qty, ord_amt], ord_result)
            usr_btn.click(debug_update_user, [usr_uid, usr_city, usr_age], usr_result)
            # 数据预览刷新
            refresh_data_btn.click(debug_list_products_df, outputs=product_table)
            refresh_data_btn.click(debug_list_orders_df, outputs=order_table)
            refresh_data_btn.click(debug_list_users_df, outputs=user_table)

        # ════════════════════════════════════════════════════════
        # Tab 3: Monitor + Charts
        # ════════════════════════════════════════════════════════
        with gr.Tab("📊 任务监控"):
            gr.Markdown("查看自主分析任务状态、Agent 思考链路，以及数据可视化图表。")

            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新任务列表")
            task_table = gr.Dataframe(label="任务列表", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    retry_id = gr.Number(label="任务ID", value=1, precision=0)
                    retry_btn = gr.Button("重试任务")
                    ignore_btn = gr.Button("忽略任务")
                    task_action_msg = gr.Textbox(label="结果", interactive=False)

            # ── 任务详情(三阶段多 Agent 报告)──
            with gr.Row():
                detail_id = gr.Number(label="任务ID(查看详情)", value=1, precision=0)
                detail_btn = gr.Button("查看任务详情", variant="secondary")
            task_detail_display = gr.Markdown("_输入任务 ID 查看报告(HIGH 事件为 Monitor→Analysis→Strategy 三段式)_")

            def show_task_detail(task_id):
                try:
                    import json as _json
                    data = _get(f"/tasks/{int(task_id)}")
                    stages = data.get("stage_results")
                    if isinstance(stages, str):
                        try:
                            stages = _json.loads(stages)
                        except Exception:
                            stages = None
                    if stages and stages.get("monitor"):
                        return "\n".join([
                            "## 🔭 Monitor 监测报告", "", stages.get("monitor", ""),
                            "", "---", "",
                            "## 🔬 Analysis 根因分析", "", stages.get("analysis", ""),
                            "", "---", "",
                            "## 💡 Strategy 运营策略", "", stages.get("strategy", ""),
                        ])
                    rt = data.get("result_text") or "_(任务尚无结果)_"
                    return f"### 任务 {task_id}(单 Agent 路径)\n\n{rt[:4000]}"
                except Exception as e:
                    return f"_加载失败: {e}_"

            detail_btn.click(show_task_detail, detail_id, task_detail_display)

            gr.Markdown("### 数据可视化")
            with gr.Row():
                rfm_plot = gr.Plot(label="RFM 分群对比")
                pie_plot = gr.Plot(label="分群占比")
            with gr.Row():
                rfm_desc = gr.Markdown()
                pie_desc = gr.Markdown()
            with gr.Row():
                radar_plot = gr.Plot(label="分群雷达图")
                flow_plot = gr.Plot(label="流转分布")
            with gr.Row():
                radar_desc = gr.Markdown()
                flow_desc = gr.Markdown()
            with gr.Row():
                trend_plot = gr.Plot(label="分群人数时间趋势")
                trend_desc = gr.Markdown()

            gr.Markdown("### 智能运营建议")
            with gr.Row():
                suggest_btn = gr.Button("💡 生成运营建议", variant="primary")
            suggest_display = gr.Markdown(
                "_点击按钮，LLM 基于上方图表同源数据生成分群级运营建议，"
                "并经过数值核查（防编造数字）。_"
            )

            def show_suggestions():
                try:
                    data = _get("/api/suggestions")
                    if isinstance(data, dict) and "detail" in data:
                        return f"_生成失败: {data['detail']}_"
                    md = data.get("suggestions") or "_(无内容)_"
                    if not data.get("grounded", False):
                        n = data.get("violations", 0)
                        md += f"\n\n> ⚠️ 建议文本中有 {n} 处数字未通过核查，请以上方图表数据为准。"
                    return md
                except Exception as e:
                    return f"_生成失败: {e}_"

            refresh_btn.click(monitor_list_tasks, outputs=task_table)
            refresh_btn.click(chart_rfm, outputs=[rfm_plot, rfm_desc])
            refresh_btn.click(chart_segment_pie, outputs=[pie_plot, pie_desc])
            refresh_btn.click(chart_radar, outputs=[radar_plot, radar_desc])
            refresh_btn.click(chart_flow, outputs=[flow_plot, flow_desc])
            refresh_btn.click(chart_trend, outputs=[trend_plot, trend_desc])
            suggest_btn.click(show_suggestions, outputs=suggest_display)
            retry_btn.click(monitor_retry_task, retry_id, task_action_msg)
            ignore_btn.click(monitor_ignore_task, retry_id, task_action_msg)

            # ── 导出 Excel(运营一键拉数)──
            gr.Markdown("### 📤 导出 Excel")
            with gr.Row():
                export_stats_btn = gr.Button("导出分群统计")
                export_funnel_btn = gr.Button("导出转化漏斗(7天)")
                export_trend_btn = gr.Button("导出分群趋势")
                export_tasks_btn = gr.Button("导出任务列表")
            export_file = gr.File(label="下载", interactive=False)
            export_msg = gr.Markdown("")

            def do_export(dataset, days=None):
                path = export_excel_file(dataset, days)
                if path and not path.startswith("_"):
                    return path, f"✅ 已生成: `{os.path.basename(path)}`"
                return None, path or "_导出失败_"

            export_stats_btn.click(
                lambda: do_export("segment_stats"), outputs=[export_file, export_msg])
            export_funnel_btn.click(
                lambda: do_export("funnel", days=7), outputs=[export_file, export_msg])
            export_trend_btn.click(
                lambda: do_export("segment_trend"), outputs=[export_file, export_msg])
            export_tasks_btn.click(
                lambda: do_export("tasks"), outputs=[export_file, export_msg])

        # ════════════════════════════════════════════════════════
        # Tab 4: 自动周报(每周一生成,可手动触发)
        # ════════════════════════════════════════════════════════
        with gr.Tab("📅 周报"):
            gr.Markdown("每周一 09:00 自动生成(错过自动补做):本周事件 + 分群趋势 + 运营建议,可直接复制分享。")

            with gr.Row():
                report_refresh_btn = gr.Button("🔄 刷新/生成周报", variant="primary")
                report_force_btn = gr.Button("♻️ 强制重新生成")
            report_display = gr.Markdown(
                "_点击按钮生成周报(幂等:同周已生成则直接显示)。_"
            )

            def show_weekly_report(force: bool = False):
                try:
                    if force:
                        data = _post("/debug/trigger-weekly-report", {"force": True})
                    else:
                        data = _get("/debug/weekly-report")
                    if isinstance(data, dict) and "detail" in data:
                        return f"_生成失败: {data['detail']}_"
                    reports = data.get("reports") or []
                    if reports:
                        content = reports[0].get("content", "")
                        if content:
                            return content
                    # 无周报 → 立即触发生成
                    gen = _post("/debug/trigger-weekly-report", {})
                    if isinstance(gen, dict) and "detail" in gen:
                        return f"_生成失败: {gen['detail']}_"
                    return gen.get("content") or "_(生成失败: 无内容)_"
                except Exception as e:
                    return f"_生成失败: {e}_"

            report_refresh_btn.click(
                lambda: show_weekly_report(force=False), outputs=report_display)
            report_force_btn.click(
                lambda: show_weekly_report(force=True), outputs=report_display)

            # ── 周报导出 Excel(多 sheet:事件/分群/高价值/建议)──
            with gr.Row():
                report_excel_btn = gr.Button("📥 导出周报 Excel")
            report_excel_file = gr.File(label="周报下载", interactive=False)
            report_excel_msg = gr.Markdown("")

            def do_export_report():
                path = export_excel_file("weekly_report")
                if path and not path.startswith("_"):
                    return path, f"✅ 已生成: `{os.path.basename(path)}`"
                return None, path or "_导出失败_"

            report_excel_btn.click(
                do_export_report, outputs=[report_excel_file, report_excel_msg])

        # ════════════════════════════════════════════════════════
        # Tab 5: LLM Observability
        # ════════════════════════════════════════════════════════
        with gr.Tab("📈 可观测性"):
            gr.Markdown("实时监控 LLM 调用统计、延迟分布和工具调用成功率。")

            obs_refresh_btn = gr.Button("🔄 刷新统计", variant="primary")

            with gr.Row():
                obs_req = gr.Number(label="总请求数", value=0, precision=0)
                obs_tok = gr.Number(label="总 Token 消耗", value=0, precision=0)
                obs_tools = gr.Number(label="工具调用次数", value=0, precision=0)
                obs_errs = gr.Number(label="错误数", value=0, precision=0)

            with gr.Row():
                obs_avg_lat = gr.Textbox(label="平均延迟", value="—")
                obs_p50 = gr.Textbox(label="P50 延迟", value="—")
                obs_p95 = gr.Textbox(label="P95 延迟", value="—")
                obs_p99 = gr.Textbox(label="P99 延迟", value="—")

            with gr.Row():
                obs_tool_rate = gr.Textbox(label="工具调用率", value="—")
                obs_fc_pass = gr.Textbox(label="事实核查通过率", value="—")
                obs_avg_tok = gr.Textbox(label="平均 Token/请求", value="—")
                obs_uptime = gr.Textbox(label="运行时间", value="—")

            gr.Markdown("### 最近请求")
            obs_recent = gr.Dataframe(
                label="最近20条请求",
                headers=["会话", "问题", "延迟ms", "Token", "工具", "核查", "错误"],
                interactive=False,
            )

            def refresh_observability():
                import pandas as pd
                try:
                    stats = _get("/stats/observability")
                    if not stats or stats.get("total_requests", 0) == 0:
                        return (0, 0, 0, 0, "—", "—", "—", "—", "—", "—", "—", "—",
                                pd.DataFrame({"提示": ["暂无数据，先发送几个问题吧"]}))

                    recent_rows = []
                    for r in stats.get("recent_requests", []):
                        recent_rows.append({
                            "会话": r["session"],
                            "问题": r["question"][:40],
                            "延迟ms": r["duration_ms"],
                            "Token": r["tokens"],
                            "工具": ",".join(r.get("tools", [])[:2]),
                            "核查": "pass" if r.get("fact_ok") else ("fail" if r.get("fact_ok") is False else "—"),
                            "错误": r.get("error", "")[:30],
                        })
                    df = pd.DataFrame(recent_rows) if recent_rows else pd.DataFrame({
                        "提示": ["暂无数据"]
                    })

                    return (
                        stats["total_requests"],
                        stats["total_tokens"],
                        stats["total_tool_calls"],
                        stats["error_count"],
                        f'{stats["avg_latency_ms"]:.0f}ms',
                        f'{stats["p50_latency_ms"]:.0f}ms',
                        f'{stats["p95_latency_ms"]:.0f}ms',
                        f'{stats["p99_latency_ms"]:.0f}ms',
                        f'{stats["tool_call_rate"]:.0%}',
                        f'{stats["fact_check_pass_rate"]:.0%}',
                        f'{stats["avg_tokens_per_request"]:.0f}',
                        f'{stats["uptime_seconds"]:.0f}s',
                        df,
                    )
                except Exception as e:
                    return (0, 0, 0, 0, "—", "—", "—", "—", "—", "—", "—", "—",
                            pd.DataFrame({"错误": [str(e)]}))

            obs_refresh_btn.click(
                refresh_observability,
                outputs=[
                    obs_req, obs_tok, obs_tools, obs_errs,
                    obs_avg_lat, obs_p50, obs_p95, obs_p99,
                    obs_tool_rate, obs_fc_pass, obs_avg_tok, obs_uptime,
                    obs_recent,
                ],
            )

        # ════════════════════════════════════════════════════════
        # Tab 5: Agent Trace
        # ════════════════════════════════════════════════════════
        with gr.Tab("🔍 Agent Trace"):
            gr.Markdown("查看 Agent 的完整思考链路 — 每个节点的耗时、工具调用、事实核查结果。")
            gr.Markdown("> 💡 先在「AI 对话」Tab 中发送一个问题，然后回到这里查看 Trace。")

            with gr.Row():
                trace_refresh_btn = gr.Button("🔄 刷新 Trace 列表", variant="primary")
                trace_clear_btn = gr.Button("🗑️ 清空 Trace", variant="stop", size="sm")

            with gr.Row():
                trace_selector = gr.Dropdown(
                    label="选择 Trace（按会话ID）",
                    choices=[],
                    interactive=True,
                    scale=3,
                )

            trace_display = gr.Markdown("_点击「刷新 Trace 列表」加载最近的 Agent 执行记录_")

            def refresh_trace_list():
                try:
                    data = _get("/traces")
                    traces = data.get("traces", [])
                    choices = [
                        f"{t['session_id']} | {t['question'][:50]} | {t['total_duration_ms']:.0f}ms"
                        for t in traces
                    ]
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)
                except Exception:
                    return gr.Dropdown(choices=[], value=None)

            def show_trace(selected: str):
                if not selected:
                    return "_选择一个 Trace 查看详情_"
                try:
                    session_id = selected.split(" | ")[0]
                    data = _get(f"/traces/{session_id}")
                    # Build markdown from trace dict
                    spans = data.get("spans", [])
                    lines = [
                        "## 🔍 Agent 执行链路",
                        "",
                        f"**Trace ID**: `{data.get('trace_id','')}`  |  **Session**: `{data.get('session_id','')}`",
                        f"**总耗时**: {data.get('total_duration_ms',0):.0f}ms  |  **总Token**: {data.get('total_tokens',0)}",
                        f"**问题**: {data.get('question','')}",
                        "",
                        "| 步骤 | 节点 | 耗时 | Token | 详情 |",
                        "|------|------|------|-------|------|",
                    ]
                    emoji_map = {
                        "preprocess": "🔍", "llm_decide": "🤖", "tools": "🔧",
                        "reflect": "🪞", "respond": "💬", "fact_check": "✅",
                    }
                    for i, s in enumerate(spans, 1):
                        emoji = emoji_map.get(s.get("node", ""), "➡️")
                        detail = s.get("output", "")[:80]
                        meta = s.get("meta", {})
                        if "tool_calls" in meta:
                            detail = f"调用: {', '.join(meta['tool_calls'])}"
                        elif meta.get("intent"):
                            detail = f"意图: {meta['intent']}"
                        elif meta.get("violations") is not None:
                            detail = f"违规: {meta['violations']}个"
                        lines.append(
                            f"| {emoji} | **{s.get('node','')}** | {s.get('duration_ms',0):.0f}ms | "
                            f"{s.get('tokens',0)} | {detail} |"
                        )
                    lines.append("")
                    if data.get("fact_check_passed") is True:
                        lines.append("✅ **事实核查通过**")
                    elif data.get("fact_check_passed") is False:
                        lines.append(f"⚠️ **事实核查未通过** — {data.get('violation_count',0)} 个违规")
                    return "\n".join(lines)
                except Exception as e:
                    return f"_加载失败: {e}_"

            def clear_all_traces():
                try:
                    _delete("/traces")
                except Exception as e:
                    logger.warning("trace_clear_failed", extra={"error": str(e)})
                return gr.Dropdown(choices=[], value=None), "_已清空_"

            trace_refresh_btn.click(refresh_trace_list, outputs=trace_selector)
            trace_selector.change(show_trace, trace_selector, trace_display)
            trace_clear_btn.click(clear_all_traces, outputs=[trace_selector, trace_display])

    return demo


demo = create_ui()

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
