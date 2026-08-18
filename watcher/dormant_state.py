"""已上报流失预警(dormant)用户集合的持久化 — 防重复触发。

主闸语义:只有"新进入 30 天未下单"的高价值用户(未在上报集合中)才触发事件,
同一批 dormant 用户不会反复触发。集合持久化到 JSON 文件,进程重启不重报。

并发:RLock 保护读-改-写;原子写(临时文件 + rename),崩溃不损坏文件。
"""

import json
import os
import threading
from typing import Iterable

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

_lock = threading.RLock()
_FILE_NAME = "dormant_reported.json"


def _path() -> str:
    settings = get_settings()
    return os.path.join(settings.CACHE_DIR, _FILE_NAME)


def load_reported() -> set[int]:
    """读取已上报用户集合(文件缺失/损坏 → 空集,不阻塞主链路)。"""
    with _lock:
        path = _path()
        try:
            if not os.path.exists(path):
                return set()
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {int(x) for x in data} if isinstance(data, list) else set()
        except Exception as exc:
            logger.warning("dormant_state_load_failed", extra={
                "path": path, "error": str(exc)[:200],
            })
            return set()


def mark_reported(ids: Iterable[int]) -> None:
    """把用户 ID 合并写入已上报集合(读-改-写全在锁内,原子落盘)。"""
    with _lock:
        current = load_reported()
        current.update(int(i) for i in ids if i is not None)
        path = _path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(sorted(current), f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("dormant_state_save_failed", extra={
                "path": path, "error": str(exc)[:200],
            })
