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
        self._last_fingerprint: str | None = None   # 数据指纹(增量重算:None=首次全量)

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

        # 增量重算:数据指纹不变 → 复用缓存(TTL 兜底,不强制重算);
        # 指纹变化(商城下单/CSV 更新/换数据源)→ force_refresh 全量重算
        try:
            from pipeline.data_loader import data_fingerprint
            from skills.user_segment import _load_and_process
            fp = data_fingerprint()
            refresh = (fp != self._last_fingerprint)
            rfm, _, _ = _load_and_process(force_refresh=refresh)
            self._last_fingerprint = fp
            if refresh:
                logger.info("poll_full_refresh", extra={"fingerprint": fp})
            else:
                logger.debug("poll_fingerprint_hit", extra={"fingerprint": fp})
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

        # 流失预警(用户级规则,独立于快照 diff):新增流失用户集去重(主闸)
        # + 事件级冷却(次闸,防三阶段 orchestrator 高频触发)。
        # HIGH_VALUE_DORMANT 是周期运营动作而非告警,节流不放在 should_suppress
        # (其"HIGH 永不抑制"语义有既有测试守护)。
        try:
            from watcher.events import detect_high_value_dormant
            from watcher import dormant_state
            dormant = detect_high_value_dormant(
                rfm, settings, dormant_state.load_reported())
            if dormant is not None and not self.task_manager.is_event_in_cooldown(
                    dormant.event_type.value,
                    settings.HIGH_VALUE_DORMANT_COOLDOWN_SECONDS):
                candidates.append(dormant)
        except Exception:
            logger.exception("dormant_detect_failed")

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

        # 流失预警:任务创建成功即标记已上报用户集(dispatch 前标记,防同轮重复;
        # 若进程恰在此前崩溃可能产生一次重复任务 —— 至少一次语义,恢复靠 retry UI)
        if event.event_type.value == "high_value_dormant":
            try:
                from watcher import dormant_state
                dormant_state.mark_reported(event.details.get("new_dormant_ids", []))
            except Exception:
                logger.exception("dormant_mark_failed")

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
        """Dispatch: HIGH 事件走三阶段多 Agent 流水线,其余走单 Agent。

        Orchestrator(Monitor→Analysis→Strategy)只对高优先级业务异常启用——
        普通波动事件单 Agent 足够,三阶段串行会放大延迟与成本。
        """
        if getattr(task_rec, "priority", "") == "high":
            await self._run_orchestrated(task_rec, query, session_id)
        else:
            await self._run_single(task_rec, query, session_id, max_rounds)

    async def _run_single(
        self, task_rec, query: str, session_id: str, max_rounds: int,
    ) -> None:
        """单 Agent 派发(原有路径):semaphore → thread pool → agent。"""
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

    async def _run_orchestrated(
        self, task_rec, query: str, session_id: str,
    ) -> None:
        """三阶段多 Agent 流水线(Monitor→Analysis→Strategy),结果三段落库。"""
        settings = get_settings()

        self.task_manager.set_running(task_rec.id)

        loop = asyncio.get_event_loop()
        async with self._llm_semaphore:
            try:
                stages: dict = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._thread_pool,
                        _run_orchestrator_sync,
                        query, session_id,
                    ),
                    timeout=settings.ORCHESTRATOR_TASK_TIMEOUT_SECONDS,
                )
                combined = (
                    f"## 🔭 监测报告\n\n{stages['monitor_result']}\n\n"
                    f"## 🔬 根因分析\n\n{stages['analysis_result']}\n\n"
                    f"## 💡 运营策略\n\n{stages['strategy_result']}"
                )
                self.task_manager.set_completed_with_stages(
                    task_rec.id, combined, 3, {
                        "trace_id": stages.get("trace_id", ""),
                        "monitor": stages["monitor_result"],
                        "analysis": stages["analysis_result"],
                        "strategy": stages["strategy_result"],
                    },
                )
                logger.info("task_orchestrated", extra={
                    "task_id": task_rec.id,
                    "trace_id": stages.get("trace_id", ""),
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
    reply, rounds, _, _ = _ask_agent_internal(query, session_id, check_mode, max_rounds)
    return reply, rounds


def _run_orchestrator_sync(query: str, session_id: str) -> dict:
    """Synchronous wrapper for the three-stage orchestrator (runs in thread).

    Returns:
        {"trace_id": str, "monitor_result": str, "analysis_result": str,
         "strategy_result": str, "elapsed_seconds": float}
    """
    from agent.orchestrator import get_orchestrator
    return get_orchestrator().run_pipeline(query)


# ── Singleton ───────────────────────────────────────────────────

_watcher_engine: WatcherEngine | None = None


def get_watcher_engine() -> WatcherEngine:
    global _watcher_engine
    if _watcher_engine is None:
        _watcher_engine = WatcherEngine()
    return _watcher_engine
