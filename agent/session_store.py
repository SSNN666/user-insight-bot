"""会话记忆持久化:JSON 文件版 LangGraph CheckpointSaver。

- 每个 thread 一个 JSON 文件(cache_data/sessions/),进程重启后多轮上下文不丢
- JsonPlusSerializer 序列化(BaseMessage/pydantic 原生支持)
- put 时按 SESSION_TTL_DAYS 做惰性清理(扫目录删过期文件)
- 单 checkpoint 语义(每 thread 只保留最新状态,不做 fork/回溯)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from log.logger import get_logger

logger = get_logger(__name__)


def _thread_key(thread_id: str) -> str:
    """thread_id → 安全文件名(session_id 可能含特殊字符)。"""
    return hashlib.md5(thread_id.encode("utf-8")).hexdigest()[:16]


class JSONCheckpointSaver(BaseCheckpointSaver):
    """文件系统 CheckpointSaver(演示级:单进程、单 checkpoint/thread)。

    用法:graph.compile(checkpointer=JSONCheckpointSaver(dir));重启后用同一目录
    继续 compile,同一 thread_id 的多轮历史自动恢复。
    """

    serde = JsonPlusSerializer()

    def __init__(self, base_dir: str, ttl_days: float = 7.0):
        super().__init__()
        os.makedirs(base_dir, exist_ok=True)
        self._dir = base_dir
        self._ttl_seconds = ttl_days * 86400
        self._lock = threading.RLock()
        self._last_sweep = 0.0

    def _path(self, thread_id: str) -> str:
        return os.path.join(self._dir, f"{_thread_key(thread_id)}.json")

    # ── JsonPlus(typed 格式)↔ JSON 文件的桥接 ──────────
    # dumps_typed 产出 ("<tag>", bytes) 对(tag 可为 msgpack/null/string 等),
    # 可能嵌套在 dict/list 深处;bytes 不可直接 json.dump → 递归 base64 包装

    @staticmethod
    def _wrap_bytes(o):
        if (isinstance(o, (list, tuple)) and len(o) == 2
                and isinstance(o[0], str) and isinstance(o[1], (bytes, bytearray))):
            return [o[0], base64.b64encode(bytes(o[1])).decode("ascii")]
        if isinstance(o, dict):
            return {k: JSONCheckpointSaver._wrap_bytes(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [JSONCheckpointSaver._wrap_bytes(v) for v in o]
        return o

    @staticmethod
    def _unwrap_bytes(o):
        # 与 _wrap_bytes 对称:["<tag>", b64串] → ("<tag>", bytes) 交回 loads_typed
        if (isinstance(o, list) and len(o) == 2
                and isinstance(o[0], str) and isinstance(o[1], str)):
            try:
                decoded = base64.b64decode(o[1], validate=True)
                if base64.b64encode(decoded).decode("ascii") == o[1]:
                    return (o[0], decoded)
            except Exception:
                pass
        if isinstance(o, dict):
            return {k: JSONCheckpointSaver._unwrap_bytes(v) for k, v in o.items()}
        if isinstance(o, list):
            return [JSONCheckpointSaver._unwrap_bytes(v) for v in o]
        return o

    def _dumps_typed(self, obj):
        return self._wrap_bytes(self.serde.dumps_typed(obj))

    def _loads_typed(self, data):
        return self.serde.loads_typed(self._unwrap_bytes(data))

    def _sweep_expired(self) -> None:
        """惰性清理过期会话文件(每 10 分钟最多扫一次)。"""
        now = time.time()
        if now - self._last_sweep < 600:
            return
        self._last_sweep = now
        try:
            for fn in os.listdir(self._dir):
                if not fn.endswith(".json"):
                    continue
                fpath = os.path.join(self._dir, fn)
                try:
                    if now - os.path.getmtime(fpath) > self._ttl_seconds:
                        os.remove(fpath)
                        logger.info("session_expired_removed", extra={"file": fn})
                except OSError:
                    pass
        except OSError:
            pass

    # ── BaseCheckpointSaver 接口 ─────────────────────────────

    def get_tuple(self, config) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        with self._lock:
            fpath = self._path(thread_id)
            if not os.path.isfile(fpath):
                return None
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                return None

        checkpoint = self._loads_typed(data["checkpoint"])
        metadata = data.get("metadata") or {}
        parent_config = None
        parent_id = checkpoint.get("parent_checkpoint_id")
        if parent_id:
            parent_config = {
                "configurable": {
                    **config.get("configurable", {}),
                    "checkpoint_id": parent_id,
                },
            }
        pending_writes = None
        if data.get("pending_writes"):
            try:
                pending_writes = [
                    (task_id, channel, self._loads_typed(value))
                    for task_id, channel, value in data["pending_writes"]
                ]
            except Exception:
                pending_writes = None
        return CheckpointTuple(config, checkpoint, metadata, parent_config,
                               pending_writes)

    def put(self, config, checkpoint, metadata, new_versions):
        thread_id = config["configurable"]["thread_id"]
        payload = {
            "checkpoint": self._dumps_typed(checkpoint),
            "metadata": metadata,
            "pending_writes": [],   # 写入新 checkpoint 后清空挂起写入
            "saved_at": time.time(),
        }
        with self._lock:
            with open(self._path(thread_id), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        self._sweep_expired()
        return {"configurable": {
            **config.get("configurable", {}),
            "checkpoint_id": checkpoint["id"],
        }}

    def put_writes(self, config, writes, task_id, task_path: str = "") -> None:
        thread_id = config["configurable"]["thread_id"]
        with self._lock:
            fpath = self._path(thread_id)
            data = {}
            if os.path.isfile(fpath):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, json.JSONDecodeError):
                    data = {}
            pending = data.get("pending_writes") or []
            for channel, value in writes:
                pending.append([task_id, channel, self._dumps_typed(value)])
            data["pending_writes"] = pending
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

    def list(self, config=None, *, filter=None, before=None, limit=None):
        """列出各 thread 的最新 checkpoint(thread 列表场景用)。"""
        try:
            fnames = sorted(
                fn for fn in os.listdir(self._dir) if fn.endswith(".json")
            )
        except OSError:
            return
        count = 0
        for fn in fnames:
            fpath = os.path.join(self._dir, fn)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                checkpoint = self._loads_typed(data["checkpoint"])
            except (OSError, json.JSONDecodeError, KeyError, Exception):
                continue
            thread_id = checkpoint.get("id") or fn
            cfg = {"configurable": {"thread_id": thread_id}}
            yield CheckpointTuple(cfg, checkpoint, data.get("metadata") or {})
            count += 1
            if limit and count >= limit:
                return

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            try:
                os.remove(self._path(thread_id))
            except OSError:
                pass

    # ── 异步接口(astream 走异步 checkpoint 路径)──────────────
    # 文件 IO 极小,直接复用同步实现(锁内线程安全)

    async def aget_tuple(self, config):
        return self.get_tuple(config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path: str = ""):
        return self.put_writes(config, writes, task_id, task_path)

    async def alist(self, config=None, *, filter=None, before=None, limit=None):
        for tup in self.list(config, filter=filter, before=before, limit=limit):
            yield tup

    async def adelete_thread(self, thread_id: str):
        return self.delete_thread(thread_id)


class LazyCheckpointSaver(BaseCheckpointSaver):
    """惰性代理:首次调用时才按当前配置创建底层 saver。

    原因:模块级图在 import 时构建,而配置(会话目录)必须等运行环境就绪
    (测试的 conftest 会先重定向所有磁盘路径)——延迟到首次调用即自动适配。
    """

    serde = JsonPlusSerializer()

    def __init__(self):
        super().__init__()
        self._inner: JSONCheckpointSaver | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> JSONCheckpointSaver:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    from config.settings import get_settings
                    settings = get_settings()
                    self._inner = JSONCheckpointSaver(
                        settings.SESSION_DIR, settings.SESSION_TTL_DAYS,
                    )
                    logger.info("session_saver_created", extra={
                        "dir": settings.SESSION_DIR,
                        "ttl_days": settings.SESSION_TTL_DAYS,
                    })
        return self._inner

    def get_tuple(self, config):
        return self._ensure().get_tuple(config)

    def put(self, config, checkpoint, metadata, new_versions):
        return self._ensure().put(config, checkpoint, metadata, new_versions)

    def put_writes(self, config, writes, task_id, task_path: str = ""):
        return self._ensure().put_writes(config, writes, task_id, task_path)

    def list(self, config=None, *, filter=None, before=None, limit=None):
        return self._ensure().list(config, filter=filter, before=before, limit=limit)

    def delete_thread(self, thread_id: str):
        return self._ensure().delete_thread(thread_id)

    async def aget_tuple(self, config):
        return self._ensure().get_tuple(config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self._ensure().put(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path: str = ""):
        return self._ensure().put_writes(config, writes, task_id, task_path)

    async def alist(self, config=None, *, filter=None, before=None, limit=None):
        async for tup in self._ensure().alist(config, filter=filter,
                                              before=before, limit=limit):
            yield tup

    async def adelete_thread(self, thread_id: str):
        return self._ensure().delete_thread(thread_id)
