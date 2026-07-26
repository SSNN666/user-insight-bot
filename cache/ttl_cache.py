"""File-based TTL cache for DataFrame serialization.

Used by the three-tier fallback in the data loader: MySQL → cache → mock.
"""

import hashlib
import os
import pickle
import time

import pandas as pd

from config.settings import get_settings
from errors.exceptions import CacheError
from log.logger import get_logger

logger = get_logger(__name__)


class TTLCache:
    """File-based cache with configurable TTL. Keys are content-hashed."""

    def __init__(self, namespace: str = "data"):
        settings = get_settings()
        self.cache_dir = os.path.join(settings.CACHE_DIR, namespace)
        self.ttl = settings.CACHE_TTL_SECONDS
        os.makedirs(self.cache_dir, exist_ok=True)

    def _key_path(self, key: str) -> str:
        safe = hashlib.sha256(key.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def get(self, key: str) -> pd.DataFrame | None:
        """Return cached DataFrame, or None if missing / expired / unreadable.

        Security note: uses ``pickle.load()`` — only safe because cache files
        are written by this same process in ``set()``.  Do NOT use with
        untrusted cache directories.
        """
        path = self._key_path(key)
        if not os.path.exists(path):
            return None
        try:
            age = time.time() - os.path.getmtime(path)
            if age > self.ttl:
                os.remove(path)
                logger.info("cache_expired", extra={"key": key, "age_seconds": round(age)})
                return None
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info("cache_hit", extra={"key": key})
            return data
        except Exception as e:
            logger.warning("cache_read_error", extra={"key": key, "error": str(e)})
            return None  # degraded, not fatal

    def set(self, key: str, data: pd.DataFrame) -> None:
        """Persist DataFrame to cache. Raises CacheError on write failure."""
        path = self._key_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info("cache_set", extra={"key": key})
        except Exception as e:
            raise CacheError(
                f"Failed to write cache: {e}", context={"key": key}
            ) from e

    def invalidate(self, key: str) -> None:
        """Remove a cached entry if it exists."""
        path = self._key_path(key)
        if os.path.exists(path):
            os.remove(path)
            logger.info("cache_invalidated", extra={"cache_key": key})

    def touch(self, key: str) -> None:
        """Update mtime without rewriting data — resets TTL."""
        path = self._key_path(key)
        if os.path.exists(path):
            os.utime(path)
            logger.debug("cache_touched", extra={"cache_key": key})
