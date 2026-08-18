"""转化漏斗:浏览 → 加购 → 下单 行为事件加载与漏斗计算。

数据源(与 pipeline/data_loader 的降级链保持一致):
  1) ``DATA_SOURCE == "tianchi"`` 且 JData Action 文件存在 → 真实行为流
     (官方编码 1=浏览 2=加购 4=下单,其余类型忽略);
  2) 否则 → 从 ``load_orders_with_join()`` 的实际订单**派生**行为流:
     每笔订单同源产生"下单"行为,加购/浏览按确定性概率合成 —— 保证漏斗的
     "下单"步与分群口径的订单用户一致,且体现上游流失(含仅浏览用户)。

窗口基准日 = ``max(date) + 1 天``(与 RFM recency 口径一致)——
避免 mock/JData 数据日期与系统今天错位导致的伪结论。
"""

import threading
import time

import numpy as np
import pandas as pd

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

# 官方 JData 行为编码:1=浏览 2=加购 3=删除 4=下单 5=关注 6=点击
JDATA_ACTION_TYPES = {1: "浏览", 2: "加购", 4: "下单"}
FUNNEL_STEPS = ["浏览", "加购", "下单"]

# ── 行为事件缓存(与 skills.user_segment._load_and_process 同款 TTL)──

_cached_actions: pd.DataFrame | None = None
_cached_at: float | None = None
_cache_lock = threading.RLock()


def load_funnel_actions(force_refresh: bool = False) -> pd.DataFrame:
    """行为事件流:返回 DataFrame(user_id:int, action_type:int∈{1,2,4}, date:datetime)。

    JData 模式下读取真实 Action CSV(受限 TIANCHI_MAX_ACTIONS 行,严禁全量读 2.2GB);
    其余模式由实际订单派生。结果按 PIPELINE_CACHE_TTL 缓存,force_refresh 绕缓存。
    """
    global _cached_actions, _cached_at
    settings = get_settings()

    with _cache_lock:
        if not force_refresh and _cached_actions is not None and _cached_at is not None:
            age = time.time() - _cached_at
            if age < settings.PIPELINE_CACHE_TTL:
                logger.debug("funnel_cache_hit", extra={"age_seconds": round(age)})
                return _cached_actions
            logger.info("funnel_cache_expired", extra={"age_seconds": round(age)})
        # Double-check:另一线程刚算完 → 复用(避免惊群)
        if not force_refresh and _cached_actions is not None and _cached_at is not None:
            age = time.time() - _cached_at
            if age < 5:
                logger.debug("funnel_cache_race_avoided")
                return _cached_actions

    # 慢路径(锁外计算,允许并发读)
    actions = None
    if settings.DATA_SOURCE == "tianchi":
        try:
            actions = _load_jdata_actions()
        except Exception as exc:
            logger.warning("funnel_jdata_failed", extra={"error": str(exc)[:200]})
    if actions is None or actions.empty:
        try:
            from pipeline.data_loader import load_orders_with_join
            orders = load_orders_with_join()
        except Exception as exc:
            logger.warning("funnel_orders_failed", extra={"error": str(exc)[:200]})
            orders = None
        actions = _derive_mock_actions(orders)

    with _cache_lock:
        _cached_actions = actions
        _cached_at = time.time()
    logger.info("funnel_actions_loaded", extra={"rows": len(actions)})
    return _cached_actions


def invalidate_funnel_cache() -> None:
    """清空行为事件缓存(下次调用重新加载)。"""
    global _cached_actions, _cached_at
    with _cache_lock:
        _cached_actions = None
        _cached_at = None


# ── JData 真实行为 ──────────────────────────────────────────────


def _load_jdata_actions() -> pd.DataFrame | None:
    """从 JData Action CSV 读行为流(仅保留浏览/加购/下单)。

    复用 data_loader 的 _find_jdata_files / _read_jdata_csvs(多文件拼接 +
    utf-8/GBK 兜底 + on_bad_lines="skip");文件缺失或读取失败 → None(回退派生)。
    """
    from pipeline.data_loader import _find_jdata_files, _read_jdata_csvs

    settings = get_settings()
    found = _find_jdata_files(settings.TIANCHI_DATA_DIR)
    if not found["action"]:
        return None
    try:
        raw = _read_jdata_csvs(
            settings.TIANCHI_DATA_DIR, found["action"],
            nrows=settings.TIANCHI_MAX_ACTIONS,
        )
    except Exception as exc:
        logger.warning("funnel_jdata_read_failed", extra={"error": str(exc)[:200]})
        return None
    if raw.empty or "type" not in raw.columns or "time" not in raw.columns:
        return None

    # type 列可能是 float("4.0")/字符串 → 统一数值化再过滤
    raw["type"] = pd.to_numeric(raw["type"], errors="coerce")
    raw = raw[raw["type"].isin(JDATA_ACTION_TYPES)].copy()
    raw["user_id"] = pd.to_numeric(raw["user_id"], errors="coerce")
    raw = raw[raw["user_id"].notna()]
    raw["user_id"] = raw["user_id"].astype(int)
    raw["date"] = pd.to_datetime(raw["time"], errors="coerce")
    raw = raw[raw["date"].notna()]
    # 时间线平移:与订单数据同口径(最新 = 今天-1),保证"最近 N 天"窗口一致
    try:
        from pipeline.data_loader import jdata_date_shift
        shift = jdata_date_shift(raw["date"])
        if shift != pd.Timedelta(0):
            raw["date"] = raw["date"] + shift
    except Exception:
        pass
    return raw[["user_id", "type", "date"]].rename(columns={"type": "action_type"})


# ── Mock 行为派生(订单同源)──────────────────────────────────────


def _derive_mock_actions(orders: pd.DataFrame | None) -> pd.DataFrame:
    """从订单派生确定性行为流(纯函数,可单测;MOCK_SEED 播种,跨运行一致)。

    - 每笔订单 → 1 条 type=4(下单),时间 = 订单时间;
    - 每笔订单以 p=0.65 生成 1 条 type=2(加购),时间 = 下单前 5~600 分钟;
    - 每笔订单生成 1~3 条 type=1(浏览),时间在加购/下单之前;
    - 额外为 20% 的订单用户生成仅浏览行为(浏览未加购用户,体现上游流失)。
    """
    settings = get_settings()
    rng = np.random.default_rng(settings.MOCK_SEED)

    if orders is None or orders.empty or "order_date" not in orders.columns:
        return pd.DataFrame(columns=["user_id", "action_type", "date"])

    df = orders[["user_id", "order_date"]].copy()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df[df["user_id"].notna()]
    df["user_id"] = df["user_id"].astype(int)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df[df["order_date"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["user_id", "action_type", "date"])

    rows: list[tuple] = []
    for _, r in df.iterrows():
        uid, od = int(r["user_id"]), r["order_date"]
        # 下单行为(每笔订单恰好 1 条,保证漏斗"下单"步与订单同源)
        rows.append((uid, 4, od))
        # 加购 + 浏览(p=0.65)
        if rng.random() < 0.65:
            cart_t = od - pd.Timedelta(minutes=int(rng.integers(5, 601)))
            rows.append((uid, 2, cart_t))
            for _ in range(int(rng.integers(1, 4))):
                view_t = cart_t - pd.Timedelta(minutes=int(rng.integers(1, 1440)))
                rows.append((uid, 1, view_t))
        else:
            for _ in range(int(rng.integers(1, 3))):
                view_t = od - pd.Timedelta(minutes=int(rng.integers(5, 1440)))
                rows.append((uid, 1, view_t))

    # 上游流失用户(订单用户池之外的新 ID,否则浏览/加购人数恒 ≤ 下单人数,
    # 漏斗显示不出梯度):
    #   - 仅浏览用户:20% 订单用户数,各 1~4 条浏览;
    #   - 加购未下单用户:40% 订单用户数,各 1 条加购 + 1~2 条浏览。
    buyers = df["user_id"].drop_duplicates()
    n_view_only = max(1, int(len(buyers) * 0.2))
    n_cart_only = max(1, int(len(buyers) * 0.4))
    max_uid = int(df["user_id"].max())
    base_ts = df["order_date"].max()
    for i in range(n_view_only):
        uid = max_uid + 1 + i
        for _ in range(int(rng.integers(1, 5))):
            view_t = base_ts - pd.Timedelta(minutes=int(rng.integers(30, 4320)))
            rows.append((uid, 1, view_t))
    for i in range(n_cart_only):
        uid = max_uid + 1 + n_view_only + i
        cart_t = base_ts - pd.Timedelta(minutes=int(rng.integers(30, 4320)))
        rows.append((uid, 2, cart_t))
        for _ in range(int(rng.integers(1, 3))):
            view_t = cart_t - pd.Timedelta(minutes=int(rng.integers(5, 600)))
            rows.append((uid, 1, view_t))

    result = pd.DataFrame(rows, columns=["user_id", "action_type", "date"])
    return result.sort_values("date").reset_index(drop=True)


# ── 漏斗计算 ────────────────────────────────────────────────────


def compute_funnel(actions: pd.DataFrame | None, days: int = 7) -> list[dict]:
    """最近 days 天的浏览→加购→下单漏斗:每步去重用户数与逐级转化率。

    窗口 = [max(date) - (days-1) 天, max(date)](相对数据基准,与 RFM 口径一致)。
    conversion_rate = 本步去重用户数 / 上步去重用户数(浏览恒 1.0);上步为 0 → 0.0。
    窗口内无任何事件 → 返回 []。
    """
    if actions is None or actions.empty or "date" not in actions.columns:
        return []
    end = actions["date"].max().normalize()
    start = end - pd.Timedelta(days=days - 1)
    win = actions[(actions["date"] >= start) & (actions["date"] < end + pd.Timedelta(days=1))]
    if win.empty:
        return []

    _STEP_CODES = {step: code for code, step in JDATA_ACTION_TYPES.items()}
    counts = {
        step: win[win["action_type"] == code]["user_id"].nunique()
        for step, code in _STEP_CODES.items()
    }

    rows: list[dict] = []
    prev_n: int | None = None
    for step in FUNNEL_STEPS:
        n = int(counts[step])
        rate = 1.0 if prev_n is None else (round(n / prev_n, 4) if prev_n else 0.0)
        rows.append({"step": step, "user_count": n, "conversion_rate": rate})
        prev_n = n
    return rows
