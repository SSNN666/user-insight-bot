"""SQLite-backed task persistence and lifecycle management.

Schema: watcher_tasks table with status FSM:
  pending → running → completed | failed | timeout
  pending → ignored
  failed | timeout → pending (retry)
"""

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

STATUSES = {"pending", "running", "completed", "failed", "ignored", "timeout"}


@dataclass
class TaskRecord:
    id: int | None = None
    event_type: str = ""
    priority: str = "normal"
    virtual_query: str = ""
    status: str = "pending"
    result_text: str | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    session_id: str = ""
    tool_rounds: int = 0
    event_details: str | None = None
    stage_results: str | None = None   # JSON:orchestrator 三段结果(monitor/analysis/strategy)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "priority": self.priority,
            "virtual_query": self.virtual_query,
            "status": self.status,
            "result_text": self.result_text,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "session_id": self.session_id,
            "tool_rounds": self.tool_rounds,
            "event_details": self.event_details,
            "stage_results": self.stage_results,
        }


class TaskManager:
    """Manages watcher_tasks in a local SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watcher_tasks (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type      TEXT    NOT NULL,
                    priority        TEXT    NOT NULL DEFAULT 'normal',
                    virtual_query   TEXT    NOT NULL,
                    status          TEXT    NOT NULL DEFAULT 'pending',
                    result_text     TEXT,
                    created_at      TEXT    NOT NULL,
                    started_at      TEXT,
                    completed_at    TEXT,
                    session_id      TEXT    NOT NULL,
                    tool_rounds     INTEGER NOT NULL DEFAULT 0,
                    event_details   TEXT
                )
            """)
            # 轻量迁移:旧库补 stage_results 列(orchestrator 三段结果)
            cols = [r[1] for r in conn.execute("PRAGMA table_info(watcher_tasks)").fetchall()]
            if "stage_results" not in cols:
                conn.execute("ALTER TABLE watcher_tasks ADD COLUMN stage_results TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status "
                "ON watcher_tasks(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_type_ts "
                "ON watcher_tasks(event_type, created_at)"
            )
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> "TaskRecord":
        d = dict(row)
        return TaskRecord(
            id=d["id"], event_type=d["event_type"], priority=d["priority"],
            virtual_query=d["virtual_query"], status=d["status"],
            result_text=d.get("result_text"),
            created_at=d["created_at"], started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            session_id=d["session_id"], tool_rounds=d.get("tool_rounds", 0),
            event_details=d.get("event_details"),
            stage_results=d.get("stage_results"),
        )

    # ── CRUD ─────────────────────────────────────────────────

    def create_task(self, event: "WatcherEvent") -> TaskRecord:  # noqa: F821
        now = self._now()
        details_json = json.dumps(event.details, ensure_ascii=False) if event.details else None
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO watcher_tasks
                   (event_type, priority, virtual_query, created_at, session_id, event_details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
                 event.priority.value if hasattr(event.priority, 'value') else str(event.priority),
                 event.virtual_query, now, event.event_id[:16], details_json),
            )
            conn.commit()
            return self.get_task(cur.lastrowid)

    def get_task(self, task_id: int) -> TaskRecord | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM watcher_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def list_tasks(
        self, status: str | None = None, limit: int = 50, offset: int = 0,
    ) -> list[TaskRecord]:
        with self._get_conn() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM watcher_tasks WHERE status=? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (status, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM watcher_tasks ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ── Status transitions ──────────────────────────────────

    def set_running(self, task_id: int) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE watcher_tasks SET status='running', started_at=? WHERE id=? AND status='pending'",
                (self._now(), task_id),
            )
            conn.commit()

    def set_completed(self, task_id: int, result_text: str, tool_rounds: int) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE watcher_tasks SET status='completed', result_text=?, tool_rounds=?, completed_at=? WHERE id=?",
                (result_text, tool_rounds, self._now(), task_id),
            )
            conn.commit()

    def set_completed_with_stages(
        self, task_id: int, result_text: str, tool_rounds: int, stage_results: dict,
    ) -> None:
        """完成时附带 orchestrator 三段结果(monitor/analysis/strategy,JSON 落库)。"""
        stages_json = json.dumps(stage_results, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE watcher_tasks
                   SET status='completed', result_text=?, tool_rounds=?,
                       completed_at=?, stage_results=? WHERE id=?""",
                (result_text, tool_rounds, self._now(), stages_json, task_id),
            )
            conn.commit()

    def set_failed(self, task_id: int, error: str) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE watcher_tasks SET status='failed', result_text=?, completed_at=? WHERE id=?",
                (error, self._now(), task_id),
            )
            conn.commit()

    def set_timeout(self, task_id: int) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE watcher_tasks SET status='timeout', completed_at=? WHERE id=?",
                (self._now(), task_id),
            )
            conn.commit()

    def ignore_task(self, task_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE watcher_tasks SET status='ignored', completed_at=? WHERE id=? AND status='pending'",
                (self._now(), task_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def retry_task(self, task_id: int) -> TaskRecord | None:
        with self._get_conn() as conn:
            cur = conn.execute(
                "UPDATE watcher_tasks SET status='pending', started_at=NULL, completed_at=NULL, result_text=NULL WHERE id=? AND status IN ('failed','timeout')",
                (task_id,),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
            return self.get_task(task_id)

    # ── Query helpers ──────────────────────────────────────

    def resume_pending(self) -> list[TaskRecord]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM watcher_tasks WHERE status='pending' ORDER BY id ASC"
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def is_event_in_cooldown(self, event_type: str, cooldown_seconds: int) -> bool:
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM watcher_tasks
                   WHERE event_type = ?
                     AND created_at > datetime(?, '-' || ? || ' seconds')
                     AND status IN ('pending', 'running', 'completed')""",
                (event_type, self._now(), cooldown_seconds),
            ).fetchone()
            return row["cnt"] > 0 if row else False


# ── Singleton ───────────────────────────────────────────────

_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        settings = get_settings()
        _task_manager = TaskManager(settings.WATCHER_DB_PATH)
    return _task_manager
