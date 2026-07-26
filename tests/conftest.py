"""Shared fixtures — redirect disk I/O to temp dirs for test isolation."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Set env vars to redirect all disk paths to temp dirs."""
    tmp = tempfile.mkdtemp(prefix="test_ub_")
    for d in ["cache", "snapshots", "logs/chat", "logs/task", "milvus"]:
        os.makedirs(os.path.join(tmp, d), exist_ok=True)

    monkeypatch.setenv("CACHE_DIR", os.path.join(tmp, "cache"))
    monkeypatch.setenv("SNAPSHOT_DIR", os.path.join(tmp, "snapshots"))
    monkeypatch.setenv("LOG_DIR_USER_CHAT", os.path.join(tmp, "logs", "chat"))
    monkeypatch.setenv("LOG_DIR_AUTO_TASK", os.path.join(tmp, "logs", "task"))
    monkeypatch.setenv("FEEDBACK_DB_PATH", os.path.join(tmp, "feedback.db"))
    monkeypatch.setenv("WATCHER_DB_PATH", os.path.join(tmp, "watcher.db"))
    monkeypatch.setenv("FLYWHEEL_DB_PATH", os.path.join(tmp, "flywheel.db"))
    monkeypatch.setenv("FLYWHEEL_MILVUS_DATA_DIR", os.path.join(tmp, "milvus"))
    monkeypatch.setenv("WATCHER_ENABLED", "false")
    monkeypatch.setenv("FLYWHEEL_ENABLED", "false")
    monkeypatch.setenv("LLM_MODEL_NAME", "qwen2.5:7b")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    # Invalidate the cached Settings singleton
    from config.settings import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()
