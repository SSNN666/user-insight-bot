"""FastAPI middleware layer: logging, rate limiting, and exception handling.

Phase 5 additions: RateLimitMiddleware (sliding window + dedup),
ExceptionHandlerMiddleware (standardized JSON errors).
"""

import hashlib
import json
import time
import traceback
from collections import deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger("api.requests", log_type="user_chat")
err_logger = get_logger("api.errors", log_type="auto_task")

# ── Constants ─────────────────────────────────────────────────────
_MAX_WINDOW_ENTRIES = 10_000    # evict stale sessions when exceeded
_CLEANUP_INTERVAL = 200          # requests between periodic cleanup runs


# ── Request Logging (Phase 0) ──────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request as a structured JSON entry."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        logger.info("request", extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(elapsed * 1000, 2),
        })
        return response


# ── Rate Limiting (Phase 5) ───────────────────────────────────


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter + similar-question dedup.

    - ``/ask`` path: per-session request cap + query dedup.
    - ``/tasks/*`` paths: skip user rate limiting.

    Memory management: stale sessions (with empty windows) are periodically
    evicted to prevent unbounded growth. A hard cap on total sessions triggers
    a full sweep when exceeded.
    """

    def __init__(self, app, *args, **kwargs):
        super().__init__(app, *args, **kwargs)
        self._windows: dict[str, deque] = {}
        self._query_hashes: dict[str, deque] = {}
        self._request_count: int = 0

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = get_settings()

        # Skip if disabled
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)

        # Skip for non-/ask paths (tasks, health, etc.)
        if not request.url.path.startswith("/ask"):
            return await call_next(request)

        # ── 读 body 一次:session 分桶 + 相似问题去重共用 ──
        # 修复:此前按 X-Session-ID header 分桶,但所有客户端只传 body 的
        # session_id → 全站共享一个限流桶。现从 body 取 session(回退 header)。
        body = await request.body()
        request._body = body   # 恢复 body 供下游读取

        session_id = ""
        if body:
            try:
                session_id = json.loads(body.decode("utf-8", "ignore")).get("session_id", "") or ""
            except Exception:
                session_id = ""
        session_id = session_id or request.headers.get("X-Session-ID", "") or "default"

        # ── Sliding window check ──
        now = time.time()
        window_secs = settings.CHAT_RATE_WINDOW
        limit = settings.CHAT_RATE_LIMIT

        window = self._windows.get(session_id)
        if window is None:
            window = deque()
            self._windows[session_id] = window

        # Remove expired entries
        while window and window[0] < now - window_secs:
            window.popleft()

        if len(window) >= limit:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMITED",
                        "message": f"请求过于频繁，每 {window_secs} 秒最多 {limit} 次",
                        "retry_after_seconds": window_secs,
                    }
                },
            )

        window.append(now)
        self._request_count += 1

        # ── Periodic cleanup: evict stale sessions ──
        if self._request_count % _CLEANUP_INTERVAL == 0:
            self._cleanup_stale_sessions()

        # ── Similar query dedup ──
        if settings.SIMILAR_QUERY_DEDUP:
            query_hash = hashlib.md5(body).hexdigest() if body else ""

            if query_hash:
                hashes = self._query_hashes.get(session_id)
                if hashes is None:
                    hashes = deque()
                    self._query_hashes[session_id] = hashes

                while hashes and hashes[0][0] < now - window_secs:
                    hashes.popleft()
                if any(h == query_hash for _, h in hashes):
                    # Don't count dedup rejections toward rate limit
                    window.pop()
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": {
                                "code": "DUPLICATE_QUERY",
                                "message": "检测到重复问题，请在窗口冷却后重试",
                            }
                        },
                    )
                hashes.append((now, query_hash))

        return await call_next(request)

    def _cleanup_stale_sessions(self) -> None:
        """Evict sessions whose windows and hash queues are both empty.

        Also triggers a hard sweep if total session count exceeds the cap
        (defense against session-ID spraying attacks).
        """
        # Always clean empty sessions
        stale = [
            k for k in list(self._windows.keys())
            if not self._windows.get(k) and not self._query_hashes.get(k)
        ]
        for k in stale:
            self._windows.pop(k, None)
            self._query_hashes.pop(k, None)

        # Hard cap: if still too many, evict the oldest entries
        if len(self._windows) > _MAX_WINDOW_ENTRIES:
            # Sort by last activity (max timestamp in window) and keep most recent
            def _last_activity(sid: str) -> float:
                w = self._windows.get(sid)
                return w[-1] if w else 0.0

            sorted_sessions = sorted(
                self._windows.keys(), key=_last_activity, reverse=True,
            )
            for sid in sorted_sessions[_MAX_WINDOW_ENTRIES:]:
                self._windows.pop(sid, None)
                self._query_hashes.pop(sid, None)
            logger.warning("rate_limiter_hard_sweep", extra={
                "evicted": len(sorted_sessions) - _MAX_WINDOW_ENTRIES,
                "remaining": _MAX_WINDOW_ENTRIES,
            })


# ── Exception Handler (Phase 5) ───────────────────────────────


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Catch all unhandled exceptions → standardized JSON response."""

    ERROR_MAP: dict[type, tuple[int, str]] = {}

    @classmethod
    def _init_error_map(cls):
        if cls.ERROR_MAP:
            return
        try:
            from errors.exceptions import (
                APIError, DatabaseError, ParameterError, ComputationError,
            )
            cls.ERROR_MAP = {
                APIError: (502, "UPSTREAM_ERROR"),
                DatabaseError: (503, "DATABASE_ERROR"),
                ParameterError: (400, "INVALID_PARAMETER"),
                ComputationError: (500, "COMPUTATION_ERROR"),
            }
        except ImportError:
            cls.ERROR_MAP = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        self._init_error_map()
        try:
            return await call_next(request)
        except Exception as exc:
            # NOTE: intentionally NOT catching BaseException (SystemExit, KeyboardInterrupt)
            # — those should propagate to the server runtime.
            exc_type = type(exc)
            status_code = 500
            error_code = "INTERNAL_ERROR"
            message = "服务器内部错误"

            # Match known exception types
            for exc_cls, (code, err_code) in self.ERROR_MAP.items():
                if issubclass(exc_type, exc_cls):
                    status_code = code
                    error_code = err_code
                    message = str(exc)
                    break

            # Log full traceback
            err_logger.error("unhandled_exception", extra={
                "path": request.url.path,
                "method": request.method,
                "error_type": exc_type.__name__,
                "error": str(exc),
                "traceback": traceback.format_exc()[:2000],
            })

            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "code": error_code,
                        "message": message,
                        "detail": str(exc) if status_code != 500 else "服务器内部错误",
                    }
                },
            )


# ── Api-Key Auth (Phase: demo-grade, /debug + /tasks only) ────


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """演示级鉴权:/debug/* 与 /tasks/* 要求 X-API-Key 头。

    DEBUG_API_KEY 为空 = 关闭(本地免鉴权)。仅覆盖调试台与任务管理端点,
    商城/AI 对话/健康检查不受影响。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # 按请求读配置(而非启动时固化),便于运行时切换与测试注入
        key = get_settings().DEBUG_API_KEY
        if key and request.url.path.startswith(("/debug/", "/tasks")):
            if request.headers.get("X-API-Key", "") != key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "code": "UNAUTHORIZED",
                            "message": "缺少或错误的 X-API-Key",
                        }
                    },
                )
        return await call_next(request)
