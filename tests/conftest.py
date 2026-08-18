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
    monkeypatch.setenv("SESSION_DIR", os.path.join(tmp, "sessions"))
    monkeypatch.setenv("FLYWHEEL_MILVUS_DATA_DIR", os.path.join(tmp, "milvus"))
    monkeypatch.setenv("WATCHER_ENABLED", "false")
    monkeypatch.setenv("FLYWHEEL_ENABLED", "false")
    monkeypatch.setenv("WEEKLY_REPORT_ENABLED", "false")   # 防 TestClient lifespan 起后台任务
    monkeypatch.setenv("LLM_MODEL_NAME", "qwen2.5:7b")

    # 统一适配器:测试不触发任何真实 LLM 调用(云 Key 置空 + 禁用本地兜底
    # → 若被测代码误调 LLM,降级链为空,优雅返回兜底文案而非发起网络请求)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("QIANFAN_API_KEY", "")
    monkeypatch.setenv("LLM_LOCAL_FALLBACK_ENABLED", "false")

    # 百度内容审核:测试走 NullCensor 直通(避免真实网络调用)
    monkeypatch.setenv("BAIDU_AK", "")
    monkeypatch.setenv("BAIDU_SK", "")
    monkeypatch.setenv("DEBUG_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    # Invalidate the cached Settings singleton
    from config.settings import get_settings
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()
