"""Feedback persistence — thumbs-up / thumbs-down with SQLite."""

import os
import sqlite3
from datetime import datetime, timezone

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


def _get_conn() -> sqlite3.Connection:
    settings = get_settings()
    db_path = settings.FEEDBACK_DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                reply TEXT NOT NULL,
                rating TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


# Init on first use, not at import time
_initialized = False


def _ensure_db() -> None:
    global _initialized
    if not _initialized:
        _init_db()
        _initialized = True


def save_feedback(
    question: str, reply: str, rating: str, session_id: str,
) -> int:
    """Save a feedback entry. Returns the new row ID."""
    _ensure_db()
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (question, reply, rating, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (question[:2000], reply[:5000], rating, session_id, now),
        )
        conn.commit()
        logger.info("feedback_saved", extra={"rating": rating})
        return cur.lastrowid


def get_feedback_stats() -> dict:
    """Return up/down counts."""
    _ensure_db()
    with _get_conn() as conn:
        up = conn.execute(
            "SELECT COUNT(*) as c FROM feedback WHERE rating='up'"
        ).fetchone()["c"]
        down = conn.execute(
            "SELECT COUNT(*) as c FROM feedback WHERE rating='down'"
        ).fetchone()["c"]
    return {"up": up, "down": down, "total": up + down}


def get_feedback_history(limit: int = 20) -> list[dict]:
    """Return recent feedback entries."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, question, rating, session_id, created_at FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
