"""Typed exception hierarchy for the user insight bot.

Four categories: Database, Computation, Parameter, API.
Every exception carries an optional `context: dict` for structured logging.
"""

from typing import Any


class UserInsightBotError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.context = context or {}


class DatabaseError(UserInsightBotError):
    """Raised for database connection, query, or timeout failures."""


class ComputationError(UserInsightBotError):
    """Raised for pandas/numpy/sklearn computation failures."""


class ParameterError(UserInsightBotError):
    """Raised for invalid inputs or validation failures."""


class APIError(UserInsightBotError):
    """Raised for LLM API errors or network errors during LLM calls."""


class CacheError(UserInsightBotError):
    """Raised for cache read/write failures (non-fatal; triggers next tier)."""
