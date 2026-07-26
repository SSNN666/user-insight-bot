"""Orchestrator: poll → detect → classify → dispatch.

Runs as a background asyncio task managed by the FastAPI lifespan.
Phase 5: independent thread pool + LLM concurrency semaphore.

Lazy init: the thread pool and semaphore are created when the engine
first starts (not at import time) and properly torn down on stop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from config.settings import get_settings
from watcher.task_manager import get_task_manager
from watcher.events import (
    compute_snapshot_diff, detect_events,
    should_suppress, merge_low_priority,
)
from pipeline.user_segmentation import load_snapshots
from log.logger import get_logger

logger = get_logger(__name__)


class WatcherEngine:
    """Background polling engine for event-driven autonomous analysis.

    Owns its thread pool and concurrency semaphore — both are created
    on ``start()`` and shut down on ``stop()``.
    """

    def __init__(self):
        self.task_manager = get_task_manager()
        self.stop_event = asyncio.Event()
        self._running = False
        self._thread_pool: ThreadPoolExecutor | None = None
        self._llm_semaphore: asyncio.Semaphore | None = None

    async def start(self) -> None:
        """Resume pending tasks, then enter the polling loop."""
        settings = get_settings()
        self._running = True
        self.stop_event.clear()

        # Lazy-init resources (not at import time)
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=settings.AUTO_TASK_POOL_SIZE,
                thread_name_prefix="auto-task",
            )
        if self._llm_semaphore is None:
            self._llm_semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY_LIMIT)

        # Resume pending tasks from prior restart
        pending = self.task_manager.resume_pending()
        if pending:
            logger.info("resuming_pending_tasks", extra={"count": len(pending)})
            for task_rec in pending:
                asyncio.create_task(self._dispatch_task_record(task_rec))

        logger.info("watcher_started", extra={
            "poll_interval_s": settings.POLL_INTERVAL_SECONDS,
        })

        # Polling loop
        while not self.stop_event.is_set():
            try:
                await self.poll_once()
            except Exception:
                logger.exception("poll_cycle_error")

            try:
                await asyncio.wait_for(
                    self.stop_event.wait(),
                    timeout=settings.POLL_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass  # interval elapsed → next cycle

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the polling loop to exit and shut down resources.

        Args:
            timeout: Max seconds to wait for in-flight agent tasks to complete.
                     Tasks exceeding this are abandoned (daemon threads die with
                     the process anyway).
        """
        self.stop_event.set()
        self._running = False
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True, cancel_futures=False)
            self._thread_pool = None
            logger.info("watcher_thread_pool_shutdown")

    async def poll_once(self) -> list:
        """Execute one full poll cycle. Returns newly created task records."""
        settings = get_settings()

        # Force-refresh pipeline → generates a new snapshot as side effect
        try:
            from skills.user_segment import _load_and_process
            _load_and_process(force_refresh=True)
        except Exception:
            logger.exception("poll_refresh_failed")
            return []

        # Load two most recent snapshots
        try:
            snaps = load_snapshots()
        except Exception:
            logger.warning("poll_load_snapshots_failed")
            return []
        if len(snaps) < 2:
            logger.debug("poll_insufficient_snapshots", extra={"count": len(snaps)})
            return []

        prev = snaps[-2]
        curr = snaps[-1]

        # Compute diff and detect events
        diff = compute_snapshot_diff(prev, curr)
        candidates = detect_events(diff, prev, curr)
        if not candidates:
            logger.debug("poll_no_events")
            return []

        # Dedup: suppress NORMAL events in cooldown
        survivors = []
        for e in candidates:
            if should_suppress(e, self.task_manager, settings.EVENT_COOLDOWN_SECONDS):
                logger.info("event_suppressed_cooldown", extra={
                    "event_type": e.event_type.value,
                })
            else:
                survivors.append(e)

        if not survivors:
            return []

        # Merge low-priority within window
        final_events = merge_low_priority(survivors, settings.EVENT_MERGE_WINDOW)
        if len(final_events) < len(survivors):
            logger.info("events_merged", extra={
                "before": len(survivors), "after": len(final_events),
            })

        # Dispatch each surviving event
        tasks = []
        for event in final_events:
            task_rec = await self._dispatch(event)
            tasks.append(task_rec)

        return tasks

    async def _dispatch(self, event) -> "TaskRecord":  # noqa: F821
        """Create task, invoke agent, finalize (shared code path)."""
        from watcher.events import WatcherEvent
        settings = get_settings()

        task_rec = self.task_manager.create_task(event)
        logger.info("task_dispatched", extra={
            "task_id": task_rec.id, "event_type": event.event_type.value,
        })

        await self._run_task(
            task_rec,
            event.virtual_query,
            f"auto-{event.event_id[:12]}",
            settings.MAX_AUTO_TOOL_ROUNDS,
        )
        return self.task_manager.get_task(task_rec.id)

    async def _dispatch_task_record(self, task_rec) -> None:
        """Re-dispatch a pending task (restart recovery)."""
        settings = get_settings()
        await self._run_task(
            task_rec,
            task_rec.virtual_query,
            task_rec.session_id,
            settings.MAX_AUTO_TOOL_ROUNDS,
        )

    async def _run_task(
        self, task_rec, query: str, session_id: str, max_rounds: int,
    ) -> None:
        """Shared dispatch implementation: semaphore → thread pool → agent."""
        settings = get_settings()

        self.task_manager.set_running(task_rec.id)

        loop = asyncio.get_event_loop()
        async with self._llm_semaphore:
            try:
                reply, rounds = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._thread_pool,
                        _invoke_agent_sync,
                        query, session_id, "strict", max_rounds,
                    ),
                    timeout=settings.AUTO_TASK_TIMEOUT_SECONDS,
                )
                self.task_manager.set_completed(task_rec.id, reply, rounds)
                logger.info("task_completed", extra={
                    "task_id": task_rec.id, "rounds": rounds,
                })
            except asyncio.TimeoutError:
                self.task_manager.set_timeout(task_rec.id)
                logger.warning("task_timeout", extra={"task_id": task_rec.id})
            except Exception as exc:
                self.task_manager.set_failed(task_rec.id, str(exc))
                logger.error("task_failed", extra={
                    "task_id": task_rec.id, "error": str(exc),
                })


def _invoke_agent_sync(
    query: str, session_id: str, check_mode: str, max_rounds: int,
) -> tuple[str, int]:
    """Synchronous wrapper for the agent invocation (runs in thread)."""
    from agent.agent import _ask_agent_internal
    return _ask_agent_internal(query, session_id, check_mode, max_rounds)


# ── Singleton ───────────────────────────────────────────────────

_watcher_engine: WatcherEngine | None = None


def get_watcher_engine() -> WatcherEngine:
    global _watcher_engine
    if _watcher_engine is None:
        _watcher_engine = WatcherEngine()
    return _watcher_engine
