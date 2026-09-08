"""FastAPI application factory."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings

# Windows 事件循环策略:psycopg(async) 不兼容默认 ProactorEventLoop,
# 仅 SESSION_STORE=postgres 时切 SelectorEventLoop(生产 Linux 无此问题,
# 默认行为完全不变——策略在 uvicorn 建环前于模块 import 时生效)
if os.name == "nt" and get_settings().SESSION_STORE == "postgres":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
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

    # 会话存储预热:SESSION_STORE=postgres 时启动即建连/建表,
    # 连不上直接抛错阻止启动(fail-fast——状态不静默降级,宁缺勿假)
    try:
        from agent.session_store import LazyCheckpointSaver
        LazyCheckpointSaver().warmup()
    except RuntimeError as e:
        get_logger("api.lifespan").critical("session_store_fail_fast", extra={
            "error": str(e),
        })
        raise

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

    # Middleware order — Starlette's add_middleware is insert(0), so the LAST
    # one registered is the OUTERMOST.  Register inner → outer:
    # 5. CORS — outermost, so short-circuited responses (429/401/403) also
    #           carry CORS headers (Vue dev server at :5173 relies on this)
    # 4. ExceptionHandler — catch all errors → JSON
    # 3. RequestLogger — structured JSON logging (sees rate-limited requests)
    # 2. RateLimiter — sliding window + dedup for /ask
    # 1. ApiKey — demo-grade auth for /debug/* + /tasks/*
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ExceptionHandlerMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    # Phase 4: watcher task management API
    from watcher.routes import router as watcher_router
    app.include_router(watcher_router)

    # Phase 10: e-commerce API (Vue frontend)
    from api.ecommerce import router as ecommerce_router
    app.include_router(ecommerce_router)

    return app


app = create_app()
