"""LLM 成本与运行观测(轻量可观测层)。

采集两类事件,SQLite 落盘(cache_data/llm_metrics.db,WAL):
- usage_events: 每轮 Agent 运行按角色落一行(来源 llm/bridge 的 per-run
  ContextVar bucket,UsageInfo 自带 model/provider/latency —— 零侵入共享
  adapter,康养项目的复制版不受影响)
- tier_events:  数据降级链每轮实际选中的层(MySQL/cache/tianchi/mock)

成本 = 按 (provider, model) 单价表估算(人民币/百万 token,2026-09 参考价,
云厂商随时调价——面板标注"估算模型";Ollama 本地免费)。估算逻辑集中在此
模块,单价表配置化,不散落在调用方。
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

# ── 单价表(人民币 / 1M tokens,估算参考价,标注更新日期)──────────
# (provider, model) → (input_rmb_per_1m, output_rmb_per_1m)
# 调价/新增模型时只改这里;ollama 不在此表 = 免费
PRICING: dict[tuple[str, str], tuple[float, float]] = {
    ("dashscope", "qwen3.8-max"):     (2.40, 9.60),
    ("dashscope", "qwen3.7-flash"):   (0.30, 1.20),
    ("dashscope", "qwen3-vl-plus"):   (1.60, 6.40),
    ("deepseek", "deepseek-chat"):    (2.00, 8.00),
    ("qianfan", "ernie-4.5-turbo-128k"): (0.80, 2.00),
}
PRICING_AS_OF = "2026-09(参考价,面板标注估算)"


def estimate_cost(provider: str, model: str, prompt_tokens: int,
                  completion_tokens: int) -> float:
    """按单价表估算单次调用成本(元);未收录/本地模型按 0 计。"""
    price = PRICING.get((provider, model))
    if not price:
        return 0.0
    in_p, out_p = price
    return round((prompt_tokens / 1_000_000) * in_p
                 + (completion_tokens / 1_000_000) * out_p, 6)


# ── 存储 ─────────────────────────────────────────────────────────

_lock = threading.RLock()


def _conn() -> sqlite3.Connection:
    settings = get_settings()
    db = settings.CACHE_DIR  # cache_data 由设置统一重定向(测试隔离)
    os.makedirs(db, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db, "llm_metrics.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'ask',
            role TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            est_cost REAL NOT NULL DEFAULT 0,
            degraded INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER NOT NULL DEFAULT 0
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tier_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            tier TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tier_ts ON tier_events(ts)")
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 写入入口 ─────────────────────────────────────────────────────


def record_usage_bucket(session_id: str, bucket: dict, kind: str = "ask") -> int:
    """把 bridge per-run bucket(role → list[UsageInfo])落库,返回行数。

    调用方:agent 每次运行结束(best-effort,失败不影响主链路)。
    任一 usage 的 provider ≠ 主链供应商 → 该调用标记 degraded(降级)。
    """
    rows: list[tuple] = []
    settings = get_settings()
    primary = settings.LLM_PROVIDER_PRIMARY
    ts = _now_iso()
    for role, usages in (bucket or {}).items():
        for u in usages:
            if not getattr(u, "total_tokens", 0) and not getattr(u, "provider", ""):
                continue
            est = estimate_cost(u.provider, u.model,
                                u.prompt_tokens, u.completion_tokens)
            rows.append((ts, session_id[:64], kind, role[:16],
                         u.provider[:32], u.model[:48],
                         u.prompt_tokens, u.completion_tokens, u.total_tokens,
                         est, int(u.provider != primary), u.latency_ms))
    if not rows:
        return 0
    with _lock:
        try:
            conn = _conn()
            _init_db(conn)
            conn.executemany(
                """INSERT INTO usage_events
                   (ts, session_id, kind, role, provider, model,
                    prompt_tokens, completion_tokens, total_tokens,
                    est_cost, degraded, latency_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
            conn.commit()
            conn.close()
            return len(rows)
        except Exception as e:
            logger.debug("llm_metrics_write_failed", extra={"error": str(e)[:120]})
            return 0


def record_data_tier(tier: str, rows: int) -> None:
    """记录一轮数据加载实际选中的降级层。"""
    if not tier:
        return
    with _lock:
        try:
            conn = _conn()
            _init_db(conn)
            conn.execute(
                "INSERT INTO tier_events (ts, tier, rows) VALUES (?,?,?)",
                (_now_iso(), tier[:16], int(rows)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("tier_metrics_write_failed", extra={"error": str(e)[:120]})


# ── 聚合查询(面板/API 用)────────────────────────────────────────


def usage_summary(hours: int = 168) -> dict:
    """近 N 小时聚合:总量 + 按天成本/请求 + 按模型 + 按角色 + 降级统计。"""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _lock:
        conn = _conn()
        _init_db(conn)

        def q(sql, args=()):
            return conn.execute(sql, args).fetchall()

        total = q("""SELECT COUNT(*) n, COALESCE(SUM(total_tokens),0) tk,
                            COALESCE(SUM(est_cost),0) cost,
                            COALESCE(SUM(degraded),0) deg
                     FROM usage_events WHERE ts > ?""", (since,))[0]
        by_day = q("""SELECT substr(ts,1,10) day,
                             SUM(est_cost) cost, COUNT(*) req,
                             SUM(total_tokens) tk
                      FROM usage_events WHERE ts > ?
                      GROUP BY day ORDER BY day""", (since,))
        by_model = q("""SELECT provider, model, COUNT(*) n,
                               SUM(total_tokens) tk, SUM(est_cost) cost
                        FROM usage_events WHERE ts > ?
                        GROUP BY provider, model ORDER BY cost DESC""", (since,))
        by_role = q("""SELECT role, COUNT(*) n, SUM(total_tokens) tk,
                              SUM(est_cost) cost, SUM(degraded) deg
                       FROM usage_events WHERE ts > ?
                       GROUP BY role""", (since,))
        avg_latency = q("""SELECT AVG(latency_ms) a FROM usage_events
                           WHERE ts > ? AND latency_ms > 0""", (since,))[0]["a"]
        conn.close()

    total = total if total else {"n": 0, "tk": 0, "cost": 0.0, "deg": 0}
    return {
        "window_hours": hours,
        "requests": total["n"], "tokens": total["tk"],
        "total_cost_rmb": round(total["cost"], 4),
        "degraded_requests": total["deg"],
        "degraded_ratio": round(total["deg"] / total["n"], 3) if total["n"] else 0.0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else 0,
        "by_day": [dict(r) for r in by_day],
        "by_model": [dict(r) for r in by_model],
        "by_role": [dict(r) for r in by_role],
        "pricing_as_of": PRICING_AS_OF,
    }


def tier_summary(hours: int = 168) -> list[dict]:
    """近 N 小时数据层选择分布。"""
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _lock:
        conn = _conn()
        _init_db(conn)
        rows = conn.execute(
            """SELECT tier, COUNT(*) n, SUM(rows) rows_loaded
               FROM tier_events WHERE ts > ?
               GROUP BY tier ORDER BY n DESC""", (since,)).fetchall()
        conn.close()
    return [dict(r) for r in rows]
