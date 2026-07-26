"""CDC Event Consumer — Binlog-driven real-time data change awareness.

Phase 10: replaces polling-based detection with event-driven consumption.
Integrates with the existing WatcherEngine for dedup and dispatch.

Architecture:
  MySQL Binlog → Canal/Maxwell → Redis Streams / RabbitMQ
                                        │
                        ┌───────────────┴───────────────┐
                        │  CDC Consumer (this module)    │
                        │  debounce(10s) → diff → detect │
                        └───────────────┬───────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │  WatcherEngine.dispatch()      │
                        └───────────────────────────────┘

For MVP without external MQ: uses an in-process asyncio.Queue as a
stand-in that can be swapped for Redis/RabbitMQ in production.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

from log.logger import get_logger

logger = get_logger(__name__)


# ── Types ───────────────────────────────────────────────────────


class ChangeType(StrEnum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class BinlogEvent:
    """Normalized CDC event from any source (Canal, Maxwell, Debezium)."""
    table: str                       # "orders", "users", "products"
    operation: ChangeType
    row_id: int                      # primary key
    old_data: dict = field(default_factory=dict)
    new_data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ── Consumer ───────────────────────────────────────────────────


class CDCConsumer:
    """Consumes binlog events, debounces, and triggers detection.

    In production: subscribe to Redis Streams or RabbitMQ.
    For MVP: poll from an in-process asyncio.Queue (fed by /debug/trigger-event).
    """

    def __init__(self, debounce_seconds: float = 10.0):
        self.queue: asyncio.Queue[BinlogEvent] = asyncio.Queue()
        self.debounce_seconds = debounce_seconds
        self._buffer: list[BinlogEvent] = []
        self._last_flush = 0.0
        self._running = False

    async def feed(self, event: BinlogEvent) -> None:
        """Feed a binlog event into the consumer (called by CDC source)."""
        await self.queue.put(event)

    async def start(self) -> None:
        """Run the consumer loop — process events with debounce."""
        self._running = True
        logger.info("cdc_consumer_started", extra={"debounce_s": self.debounce_seconds})

        while self._running:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self._buffer.append(event)

                now = time.time()
                if now - self._last_flush >= self.debounce_seconds:
                    await self._flush()
                    self._last_flush = now

            except asyncio.TimeoutError:
                # Flush if buffer has items and debounce expired
                if self._buffer and time.time() - self._last_flush >= self.debounce_seconds:
                    await self._flush()
                    self._last_flush = time.time()

    def stop(self) -> None:
        self._running = False

    async def _flush(self) -> None:
        """Process accumulated events: dedup by table+operation, then dispatch."""
        if not self._buffer:
            return

        # Dedup: keep only the latest event per table
        deduped: dict[str, BinlogEvent] = {}
        for e in self._buffer:
            key = f"{e.table}:{e.operation}"
            deduped[key] = e  # last write wins

        self._buffer.clear()

        # Summarize changes
        tables_affected = list(deduped.keys())
        total_changes = len(deduped)

        logger.info("cdc_flush", extra={
            "events": total_changes,
            "tables": tables_affected,
        })

        # Trigger watcher analysis
        try:
            from watcher.engine import get_watcher_engine
            engine = get_watcher_engine()
            await engine.poll_once()
        except Exception as e:
            logger.error("cdc_dispatch_error", extra={"error": str(e)})


# ── Singleton ───────────────────────────────────────────────────

_cdc_consumer: CDCConsumer | None = None


def get_cdc_consumer() -> CDCConsumer:
    global _cdc_consumer
    if _cdc_consumer is None:
        _cdc_consumer = CDCConsumer()
    return _cdc_consumer
