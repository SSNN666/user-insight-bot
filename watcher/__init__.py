"""Phase 4: Event-Driven Autonomous Agent — watcher subsystem."""

from watcher.task_manager import TaskManager, TaskRecord, get_task_manager
from watcher.events import (
    WatcherEvent, SnapshotDiff, EventType, Priority,
    compute_snapshot_diff, detect_events,
)
from watcher.engine import WatcherEngine, get_watcher_engine
from watcher.routes import router as watcher_router
