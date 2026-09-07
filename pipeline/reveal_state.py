"""滚动揭晓(rolling reveal)进度持久化 — 按墙钟日逐步露出 JData 历史窗口。

背景:JData 数据整体平移后是"刚体",每次重算分群结果不变 → 快照/Watcher
检测不到演化。滚动揭晓让数据面"每天长一天":第 1 天露出原始窗口最后
PREHEAT 天,之后每过一个墙钟日 +STEP,直到全窗口(稳态 = 动态对齐现状)。

状态文件 ``cache_data/tianchi_reveal.json``:
    {revealed_days, total_days, last_advanced}

- 按墙钟日幂等:``today > last_advanced`` 才推进 —— watcher 与 loader 两处
  调用(甚至多进程)都不会重复推进;同一天并发写出的值相同,丢更新无害;
- 时钟回拨:不满足推进条件则等待,不倒退;
- 文件缺失/损坏 → 默认值 + 警告日志,永不抛(不阻塞数据主链路)。

模板:watcher/dormant_state.py(RLock + 原子写 + 路径惰性解析)。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

_lock = threading.RLock()
_FILE_NAME = "tianchi_reveal.json"

DEFAULT_STATE: dict = {
    "revealed_days": 0,      # 已露出的原始天数(0=未初始化)
    "total_days": 0,         # 原始数据总跨度(天),CSV 变化时用于钳制
    "last_advanced": None,   # 上次推进的墙钟日 "YYYY-MM-DD"
}


def _path() -> str:
    return os.path.join(get_settings().CACHE_DIR, _FILE_NAME)


def load_state() -> dict:
    """读取揭晓进度(文件缺失/损坏 → 默认值,不抛)。"""
    with _lock:
        path = _path()
        try:
            if not os.path.exists(path):
                return dict(DEFAULT_STATE)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("state 必须是 JSON 对象")
            state = dict(DEFAULT_STATE)
            state.update({k: v for k, v in data.items() if k in DEFAULT_STATE})
            state["revealed_days"] = max(0, int(state.get("revealed_days", 0)))
            state["total_days"] = max(0, int(state.get("total_days", 0)))
            return state
        except Exception as exc:
            logger.warning("reveal_state_load_failed", extra={
                "path": path, "error": str(exc)[:200],
            })
            return dict(DEFAULT_STATE)


def _save(state: dict) -> None:
    """原子落盘(读-改-写全在锁内调用)。"""
    path = _path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("reveal_state_save_failed", extra={
            "path": path, "error": str(exc)[:200],
        })


def _settings() -> tuple[int, int]:
    settings = get_settings()
    return int(settings.TIANCHI_REVEAL_PREHEAT_DAYS), int(settings.TIANCHI_REVEAL_STEP)


def ensure_state(span_days: int, today: date | None = None) -> dict:
    """loader 侧单入口:初始化 + 跨度钳制 + 日推进,返回最新状态。

    - 未初始化(``revealed_days==0``)→ 露出 ``min(PREHEAT, span)`` 天,
      并把 ``last_advanced`` 记为今天(首日不额外推进);
    - 原始跨度收缩(CSV 换过)→ ``revealed_days`` 钳到新跨度;
    - 新墙钟日且未到顶 → ``revealed_days = min(span, revealed + STEP)``。
    有变化才落盘;任何异常降级为仅读状态,不抛。
    """
    preheat, step = _settings()
    with _lock:
        state = load_state()
        today = today or date.today()
        changed = False

        if state["revealed_days"] <= 0:
            state["revealed_days"] = max(1, min(preheat, span_days))
            state["total_days"] = span_days
            state["last_advanced"] = today.isoformat()   # 首日计数完成,不再推进
            changed = True
        else:
            if span_days <= 0:
                return state  # 数据异常,保持现状
            if state["total_days"] != span_days:
                state["total_days"] = span_days
                state["revealed_days"] = min(state["revealed_days"], span_days)
                changed = True
            last = state.get("last_advanced")
            if last is None:
                # 旧状态缺日期(半写/旧格式):只补记,当天不推进
                state["last_advanced"] = today.isoformat()
                changed = True
            elif today.isoformat() > str(last) and state["revealed_days"] < span_days:
                state["revealed_days"] = min(span_days, state["revealed_days"] + step)
                state["last_advanced"] = today.isoformat()
                changed = True

        if changed:
            _save(state)
        return state


def maybe_advance(today: date | None = None) -> bool:
    """watcher 侧:墙钟日变化则推进一天(STEP)。状态缺失/已到顶 → False。

    不初始化状态(初始化只发生在真正加载数据时,由 ensure_state 完成);
    因此本函数只读不改的调用路径(如指纹)不受影响。
    """
    with _lock:
        state = load_state()
        if state["revealed_days"] <= 0 or state["total_days"] <= 0:
            return False
        if state["revealed_days"] >= state["total_days"]:
            return False
        today = today or date.today()
        last = state.get("last_advanced")
        if last is not None and today.isoformat() <= str(last):
            return False
        _, step = _settings()
        state["revealed_days"] = min(state["total_days"], state["revealed_days"] + step)
        state["last_advanced"] = today.isoformat()
        _save(state)
        return True


def describe() -> dict:
    """debug 端点用:只读快照 + 配置镜像。"""
    state = load_state()
    settings = get_settings()
    return {
        "enabled": bool(settings.TIANCHI_REVEAL_ENABLED),
        "preheat": int(settings.TIANCHI_REVEAL_PREHEAT_DAYS),
        "step": int(settings.TIANCHI_REVEAL_STEP),
        "revealed_days": state["revealed_days"],
        "total_days": state["total_days"],
        "last_advanced": state["last_advanced"],
        "steady": state["total_days"] > 0 and state["revealed_days"] >= state["total_days"],
    }


def force_set(revealed_days: int, today: date | None = None) -> dict:
    """debug/手动端点用:直接设定揭晓进度(钳到已知跨度),并落盘。

    状态未初始化(total_days==0)时只记录目标值,待首次加载以 preheat 初始化
    —— 手动端点主要在已初始化的运行中使用,此分支极少触发。
    """
    with _lock:
        state = load_state()
        if state["total_days"] > 0:
            state["revealed_days"] = max(0, min(int(revealed_days), state["total_days"]))
        else:
            state["revealed_days"] = max(0, int(revealed_days))
        state["last_advanced"] = (today or date.today()).isoformat()
        _save(state)
        return state
