"""FastAPI application factory."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from api.routes import router
from api.middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    ExceptionHandlerMiddleware,
    ApiKeyMiddleware,
)
from log.logger import get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: launch watcher background task. Shutdown: stop watcher."""
    settings = get_settings()
    watcher_task: asyncio.Task | None = None

    if settings.WATCHER_ENABLED:
        from watcher.engine import get_watcher_engine
        engine = get_watcher_engine()
        watcher_task = asyncio.create_task(engine.start())
        get_logger("api.lifespan").info("watcher_started", extra={
            "poll_interval": settings.POLL_INTERVAL_SECONDS,
        })

    # Phase 7: Flywheel scheduler
    flywheel_task: asyncio.Task | None = None
    if settings.FLYWHEEL_ENABLED:
        from flywheel.scheduler import get_flywheel_scheduler
        flywheel_task = asyncio.create_task(get_flywheel_scheduler().start())
        get_logger("api.lifespan").info("flywheel_started", extra={
            "interval": settings.FLYWHEEL_UPDATE_INTERVAL,
        })

    # 自动周报调度器(每周一生成,错过自动补做)
    weekly_task: asyncio.Task | None = None
    if settings.WEEKLY_REPORT_ENABLED:
        from watcher.weekly_report import get_weekly_report_scheduler
        weekly_task = asyncio.create_task(get_weekly_report_scheduler().start())
        get_logger("api.lifespan").info("weekly_report_started", extra={
            "hour": settings.WEEKLY_REPORT_HOUR,
        })

    yield

    if watcher_task is not None:
        from watcher.engine import get_watcher_engine
        try:
            get_watcher_engine().stop()
            watcher_task.cancel()
            await watcher_task
        except asyncio.CancelledError:
            pass
        get_logger("api.lifespan").info("watcher_stopped")

    if flywheel_task is not None:
        from flywheel.scheduler import get_flywheel_scheduler
        try:
            get_flywheel_scheduler().stop()
            flywheel_task.cancel()
            await flywheel_task
        except asyncio.CancelledError:
            pass
        get_logger("api.lifespan").info("flywheel_stopped")

    if weekly_task is not None:
        from watcher.weekly_report import get_weekly_report_scheduler
        try:
            get_weekly_report_scheduler().stop()
            weekly_task.cancel()
            await weekly_task
        except asyncio.CancelledError:
            pass
        get_logger("api.lifespan").info("weekly_report_stopped")

    # Clean up vector store (Milvus close)
    from flywheel.vector_store import reset_vector_store
    reset_vector_store()
    get_logger("api.lifespan").info("vector_store_closed")
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # CORS — origins from settings (Vue dev server + Gradio by default)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware order (outermost first):
    # 1. ExceptionHandler — catch all errors → JSON
    # 2. ApiKey — demo-grade auth for /debug/* + /tasks/*
    # 3. RateLimiter — sliding window + dedup for /ask
    # 4. RequestLogger — structured JSON logging
    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(router)

    # Phase 4: watcher task management API
    from watcher.routes import router as watcher_router
    app.include_router(watcher_router)

    # Phase 10: e-commerce API (Vue frontend)
    from api.ecommerce import router as ecommerce_router
    app.include_router(ecommerce_router)

    return app


app = create_app()
