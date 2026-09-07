"""Multi-source sample collector — feedback, auto tasks, manual annotations."""

import os
import sqlite3

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


def _get_feedback_conn():
    settings = get_settings()
    db = settings.FEEDBACK_DB_PATH
    os.makedirs(os.path.dirname(db), exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _get_watcher_conn():
    settings = get_settings()
    db = settings.WATCHER_DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _get_flywheel_conn():
    settings = get_settings()
    db = settings.FLYWHEEL_DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def collect_from_feedback(since_id: int = 0) -> list[dict]:
    """Collect from feedback.db — entries not yet ingested."""
    samples: list[dict] = []
    try:
        with _get_feedback_conn() as conn:
            rows = conn.execute(
                "SELECT id, question, reply, rating, session_id, created_at FROM feedback WHERE id > ? ORDER BY id",
                (since_id,),
            ).fetchall()
            for r in rows:
                ext_key = f"ext_id:feedback:{r['id']}"
                samples.append({
                    "question": r["question"], "reply": r["reply"],
                    "source": "feedback", "feedback_rating": r["rating"],
                    "external_id": r["id"], "ext_key": ext_key,
                    "created_at": r["created_at"],
                })
        logger.info("collector_feedback", extra={"count": len(samples)})
    except Exception as e:
        logger.warning("collector_feedback_error", extra={"error": str(e)})
    return samples


def collect_from_auto_tasks(since_id: int = 0) -> list[dict]:
    """Collect from watcher_tasks — completed tasks with non-empty result."""
    samples: list[dict] = []
    try:
        with _get_watcher_conn() as conn:
            rows = conn.execute(
                """SELECT id, virtual_query, result_text, tool_rounds, created_at
                   FROM watcher_tasks
                   WHERE status='completed' AND result_text IS NOT NULL AND result_text != ''
                     AND id > ?
                   ORDER BY id""",
                (since_id,),
            ).fetchall()
            for r in rows:
                ext_key = f"ext_id:auto_task:{r['id']}"
                samples.append({
                    "question": r["virtual_query"], "reply": r["result_text"],
                    "source": "auto_task", "feedback_rating": None,
                    "external_id": r["id"], "ext_key": ext_key,
                    "tool_rounds": r["tool_rounds"] or 0,
                    "created_at": r["created_at"],
                })
        logger.info("collector_auto_tasks", extra={"count": len(samples)})
    except Exception as e:
        logger.warning("collector_auto_tasks_error", extra={"error": str(e)})
    return samples


def collect_from_manual(since_id: int = 0) -> list[dict]:
    """Collect from manual_annotations table (in flywheel.db)."""
    samples: list[dict] = []
    try:
        with _get_flywheel_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS manual_annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()
            rows = conn.execute(
                "SELECT id, question, reply, rating, created_at FROM manual_annotations WHERE id > ? ORDER BY id",
                (since_id,),
            ).fetchall()
            for r in rows:
                ext_key = f"ext_id:manual:{r['id']}"
                samples.append({
                    "question": r["question"], "reply": r["reply"],
                    "source": "manual", "feedback_rating": r["rating"],
                    "external_id": r["id"], "ext_key": ext_key,
                    "created_at": r["created_at"],
                })
        logger.info("collector_manual", extra={"count": len(samples)})
    except Exception as e:
        logger.warning("collector_manual_error", extra={"error": str(e)})
    return samples


def collect_all(cursors: dict[str, int] | None = None) -> list[dict]:
    """Aggregate all sources, dedup by ext_key across ALL samples (not just positives).

    cursors: per-source incremental cursors, e.g.
             {"feedback": int, "auto_task": int, "manual": int}.
             三个源的 ID 空间相互独立,必须按源分别游标;共用一个游标会在
             游标推进后永久漏采小 ID 源的新记录(id ≤ 游标且从未入库)。
             缺省从 0 开始,重复收集由 ext_key 去重保证幂等。

    Queries the sample_library directly for existing ext_keys to ensure
    negatives and manual annotations are also deduplicated.
    """
    cursors = cursors or {}
    # Query existing ext_keys directly from the DB (covers all ratings)
    existing_ids: set[str] = set()
    try:
        with _get_flywheel_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT keywords FROM sample_library WHERE keywords LIKE 'ext_id:%'"
            ).fetchall()
            existing_ids = {r["keywords"] for r in rows}
    except Exception:
        logger.debug("collector_dedup_fallback")

    all_samples = (
        collect_from_feedback(cursors.get("feedback", 0))
        + collect_from_auto_tasks(cursors.get("auto_task", 0))
        + collect_from_manual(cursors.get("manual", 0))
    )
    # Dedup
    new = [s for s in all_samples if s["ext_key"] not in existing_ids]
    logger.info("collector_all", extra={"total": len(all_samples), "new": len(new)})
    return new
