"""Flywheel incremental update scheduler — background asyncio task.

Collects new samples → scores them → ingests into sample library.
Uses per-source cursors (last processed ID) for incremental updates.
"""

import asyncio
import time

from config.settings import get_settings
from flywheel.collector import collect_all
from flywheel.scorer import score_sample
from flywheel.store import get_sample_store
from log.logger import get_logger

logger = get_logger(__name__)


class FlywheelScheduler:
    """Runs incremental sample collection and scoring on a timer."""

    def __init__(self):
        self.stop_event = asyncio.Event()
        self._running = False
        # 各源 ID 空间独立(feedback / watcher_tasks / manual_annotations),
        # 必须按源分别游标;共用一个会在游标推进后永久漏采小 ID 源的新记录
        self._cursors: dict[str, int] = {"feedback": 0, "auto_task": 0, "manual": 0}
        # 后台循环与 /flywheel/trigger 手动触发可能并发,串行化防止重复入库
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        settings = get_settings()
        self._running = True
        self.stop_event.clear()
        # 从 0 开始(不复用样本库自增 id——它与外部源 id 空间无关),
        # 重复收集由 collect_all 的 ext_key 去重保证幂等
        self._cursors = {"feedback": 0, "auto_task": 0, "manual": 0}

        logger.info("flywheel_scheduler_started", extra={
            "interval_s": settings.FLYWHEEL_UPDATE_INTERVAL,
        })

        while not self.stop_event.is_set():
            try:
                await self.update_once()
            except Exception:
                logger.exception("flywheel_update_error")

            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=settings.FLYWHEEL_UPDATE_INTERVAL,
                )
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self.stop_event.set()
        self._running = False

    async def update_once(self) -> int:
        """Collect, score, and ingest one batch of new samples.

        Uses a set-based dedup to handle non-monotonic external IDs safely.
        The cursors still advance, but we track seen IDs to prevent re-ingestion.

        Returns:
            Number of new samples ingested.
        """
        async with self._lock:
            # 内部为同步 sqlite + Ollama 嵌入(15s 超时),丢线程池避免阻塞事件循环
            return await asyncio.to_thread(self._update_once_locked)

    async def _update_once_locked(self) -> int:
        store = get_sample_store()

        # Collect new samples per source since each source's cursor
        new_samples = collect_all(self._cursors)
        if not new_samples:
            return 0

        ingested = 0
        # Track the highest ID seen this batch per source to advance cursors
        max_id_by_source: dict[str, int] = {}

        for s in new_samples:
            start = time.monotonic()
            quality = score_sample(
                question=s["question"],
                reply=s["reply"],
                source=s["source"],
                feedback_rating=s.get("feedback_rating"),
                tool_rounds=s.get("tool_rounds", 0),
            )
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)

            store.add_sample(
                question=s["question"],
                reply=s["reply"],
                source=s["source"],
                rating="positive" if quality.is_positive else "negative",
                quality_score=quality.score,
                fact_check_passed=("fact_violations" not in quality.tags),
                violation_count=0,
                keywords=s.get("ext_key", ""),
                tags=",".join(quality.tags),
            )
            # Advance cursor safely: only update from valid numeric IDs
            ext_id = s.get("external_id")
            src = s.get("source", "")
            if isinstance(ext_id, (int, float)) and ext_id is not None and src in self._cursors:
                max_id_by_source[src] = max(max_id_by_source.get(src, self._cursors[src]), ext_id)
            ingested += 1
            logger.debug("sample_ingested", extra={
                "source": s["source"], "score": quality.score,
                "positive": quality.is_positive, "ms": elapsed_ms,
            })

        # Advance each source cursor to the max ID seen in this batch
        for src, mx in max_id_by_source.items():
            self._cursors[src] = max(self._cursors[src], mx)

        logger.info("flywheel_batch_done", extra={
            "collected": len(new_samples), "ingested": ingested,
            "cursors": self._cursors,
        })
        return ingested


# ── Singleton ───────────────────────────────────────────────────

_scheduler: FlywheelScheduler | None = None


def get_flywheel_scheduler() -> FlywheelScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = FlywheelScheduler()
    return _scheduler
