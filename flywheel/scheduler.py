"""Flywheel incremental update scheduler — background asyncio task.

Collects new samples → scores them → ingests into sample library.
Uses a cursor-based approach (last processed ID) for incremental updates.
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
        self._last_ingested_id = 0

    async def start(self) -> None:
        settings = get_settings()
        self._running = True
        self.stop_event.clear()
        self._last_ingested_id = get_sample_store().get_max_ingested_id()

        logger.info("flywheel_scheduler_started", extra={
            "interval_s": settings.FLYWHEEL_UPDATE_INTERVAL,
            "last_ingested_id": self._last_ingested_id,
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
        The cursor still advances, but we track seen IDs to prevent re-ingestion.

        Returns:
            Number of new samples ingested.
        """
        store = get_sample_store()

        # Collect new samples since last cursor
        new_samples = collect_all(since_id=self._last_ingested_id)
        if not new_samples:
            return 0

        ingested = 0
        # Track the highest ID seen this batch to advance the cursor safely
        max_id_this_batch = self._last_ingested_id

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
            if isinstance(ext_id, (int, float)) and ext_id is not None:
                max_id_this_batch = max(max_id_this_batch, ext_id)
            ingested += 1
            logger.debug("sample_ingested", extra={
                "source": s["source"], "score": quality.score,
                "positive": quality.is_positive, "ms": elapsed_ms,
            })

        # Advance cursor to the max ID seen in this batch
        self._last_ingested_id = max_id_this_batch

        logger.info("flywheel_batch_done", extra={
            "collected": len(new_samples), "ingested": ingested,
            "cursor": self._last_ingested_id,
        })
        return ingested


# ── Singleton ───────────────────────────────────────────────────

_scheduler: FlywheelScheduler | None = None


def get_flywheel_scheduler() -> FlywheelScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = FlywheelScheduler()
    return _scheduler
