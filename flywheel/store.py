"""Sample library — SQLite persistence for positive/negative Q&A samples."""

import os
import sqlite3
from datetime import datetime, timezone

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)


class SampleStore:
    """CRUD for the sample_library table."""

    def __init__(self, db_path: str | None = None):
        settings = get_settings()
        self.db_path = db_path or settings.FLYWHEEL_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sample_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    source TEXT NOT NULL,
                    rating TEXT NOT NULL DEFAULT 'positive',
                    quality_score REAL DEFAULT 0.5,
                    fact_check_passed INTEGER DEFAULT 1,
                    violation_count INTEGER DEFAULT 0,
                    keywords TEXT DEFAULT '',
                    tags TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    ingested_at TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sample_rating ON sample_library(rating)"
            )
            conn.commit()

    def add_sample(
        self, question: str, reply: str, source: str, rating: str,
        quality_score: float = 0.5, fact_check_passed: bool = True,
        violation_count: int = 0, keywords: str = "", tags: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO sample_library
                   (question, reply, source, rating, quality_score, fact_check_passed,
                    violation_count, keywords, tags, created_at, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (question[:2000], reply[:5000], source, rating, quality_score,
                 int(fact_check_passed), violation_count, keywords, tags, now, now),
            )
            conn.commit()
            row_id = cur.lastrowid

        # Also insert into vector store for semantic retrieval
        if rating == "positive" and quality_score >= 0.5:
            try:
                from flywheel.vector_store import get_vector_store
                get_vector_store().insert(row_id, question, reply)
                logger.debug("vector_indexed", extra={"sample_id": row_id})
            except Exception as e:
                logger.debug("vector_index_skipped", extra={"error": str(e)})

        return row_id

    def get_positives(self, limit: int = 100) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sample_library WHERE rating='positive' ORDER BY quality_score DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_negatives(self, limit: int = 100) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sample_library WHERE rating='negative' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_by_source(self) -> dict:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT source, rating, COUNT(*) as cnt FROM sample_library GROUP BY source, rating"
            ).fetchall()
            result: dict = {}
            for r in rows:
                key = f"{r['source']}_{r['rating']}"
                result[key] = r["cnt"]
            return result

    def get_max_ingested_id(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) as mx FROM sample_library WHERE ingested_at IS NOT NULL"
            ).fetchone()
            return row["mx"] if row else 0

    def all_samples(self, limit: int = 200) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, question, reply, rating, quality_score, keywords, tags FROM sample_library WHERE rating='positive' ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]


# Singleton
_store: SampleStore | None = None


def get_sample_store() -> SampleStore:
    global _store
    if _store is None:
        _store = SampleStore()
    return _store
