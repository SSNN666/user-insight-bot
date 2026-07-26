"""Health check logic: MySQL, LLM API, and Watcher status."""

import time

from sqlalchemy import text

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

_start_time = time.time()


async def check_mysql() -> dict:
    """Test MySQL connectivity with a lightweight SELECT 1."""
    try:
        from pipeline.data_loader import get_db_engine
        engine = get_db_engine()
        start = time.monotonic()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency = round((time.monotonic() - start) * 1000, 2)
        engine.dispose()
        return {"status": "up", "latency_ms": latency}
    except Exception as e:
        return {"status": "down", "error": str(e)}


async def check_llm_api() -> dict:
    """Check LLM API reachability via a lightweight ping."""
    settings = get_settings()
    try:
        import httpx
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{settings.OPENAI_BASE_URL.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            )
        latency = round((time.monotonic() - start) * 1000, 2)
        return {
            "status": "up" if resp.status_code < 500 else "degraded",
            "model": settings.LLM_MODEL_NAME,
            "latency_ms": latency,
            "http_status": resp.status_code,
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


async def check_watcher() -> dict:
    """Check watcher engine status and pending task count."""
    try:
        from watcher.engine import get_watcher_engine
        from watcher.task_manager import get_task_manager

        engine = get_watcher_engine()
        tm = get_task_manager()
        pending = tm.resume_pending()

        return {
            "status": "up" if engine._running else "stopped",
            "running": engine._running,
            "pending_tasks": len(pending),
        }
    except Exception as e:
        return {"status": "down", "error": str(e)}


async def get_full_health() -> dict:
    """Aggregate all health checks."""
    mysql = await check_mysql()
    llm = await check_llm_api()
    watcher = await check_watcher()

    return {
        "mysql": mysql,
        "llm_api": llm,
        "watcher": watcher,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
