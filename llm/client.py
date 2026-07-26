"""Unified OpenAI-compatible LLM client with timeout, retry, and token tracking.

Wraps the /chat/completions endpoint. Handles transport-level retries
via tenacity and accumulates TokenUsage across calls.
"""

import time
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import get_settings
from errors.exceptions import APIError
from log.logger import get_logger

logger = get_logger(__name__)

# ── retry predicate ──────────────────────────────────────────────
RETRYABLE = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.HTTPStatusError,  # 5xx only — filtered in _call_api
)


def _is_retryable(exception: BaseException) -> bool:
    if isinstance(exception, httpx.HTTPStatusError):
        return 500 <= exception.response.status_code < 600
    return isinstance(exception, RETRYABLE)


# ── Token tracker ────────────────────────────────────────────────


class TokenUsage:
    """Accumulates prompt / completion token counts across calls."""

    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_calls += 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_calls": self.total_calls,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


# ── Client ───────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions endpoint.

    Uses a shared ``httpx.Client`` with connection pooling (HTTP keep-alive)
    so repeated calls don't pay the TCP handshake cost.

    Thread-safe for reads; the shared client is created once and reused.
    """

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.OPENAI_BASE_URL.rstrip("/")
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.LLM_MODEL_NAME
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.max_retries = settings.LLM_MAX_RETRIES
        self.usage = TokenUsage()

        # Shared client with connection pooling — survives across calls
        self._http = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=20,
                keepalive_expiry=30.0,
            ),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _call_api(self, messages: list[dict], **kwargs) -> dict:
        """Single API call (retried by the @retry decorator on chat())."""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),
        }
        if "extra_body" in kwargs:
            payload.update(kwargs["extra_body"])

        response = self._http.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def chat(
        self, messages: list[dict], **kwargs
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat request.

        Returns:
            (response_text, usage_dict).

        Raises:
            APIError: after all retries are exhausted.
        """
        start = time.monotonic()

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=16),
            retry=retry_if_exception_type(RETRYABLE),
            reraise=True,
        )
        def _with_retry() -> dict:
            return self._call_api(messages, **kwargs)

        try:
            data = _with_retry()
            elapsed = time.monotonic() - start

            usage_info = data.get("usage", {})
            prompt_tok = usage_info.get("prompt_tokens", 0)
            completion_tok = usage_info.get("completion_tokens", 0)
            self.usage.add(prompt_tok, completion_tok)

            choice = data["choices"][0]
            content = choice["message"]["content"]

            logger.info("llm_call", extra={
                "model": self.model,
                "duration_ms": round(elapsed * 1000, 2),
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
            })
            return content, usage_info

        except Exception as e:
            logger.error("llm_call_failed", extra={
                "model": self.model,
                "error": str(e),
                "duration_ms": round((time.monotonic() - start) * 1000, 2),
            })
            raise APIError(
                f"LLM API call failed: {e}", context={"model": self.model}
            ) from e

    def invoke(self, prompt: str) -> str:
        """Simple invoke(prompt) → text — LangChain-compatible convenience API.

        Used by the flywheel retriever and eval judge for lightweight LLM calls
        that don't need the full agent graph.
        """
        text, _ = self.chat([{"role": "user", "content": prompt}])
        return text

    def close(self) -> None:
        """Release the shared HTTP client and its connection pool."""
        self._http.close()
        logger.debug("llm_client_closed")


# ── Singleton ───────────────────────────────────────────────────

_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    """Close and reset the LLM client singleton."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# Release connections on normal process exit
import atexit
atexit.register(reset_llm_client)
