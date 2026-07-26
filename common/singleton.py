"""Lightweight, type-safe singleton factory.

Usage in any module that needs a singleton::

    from common.singleton import singleton

    @singleton
    def get_my_service() -> MyService:
        return MyService()

    # Replaces:
    # _instance: MyService | None = None
    # def get_my_service():
    #     global _instance
    #     if _instance is None:
    #         _instance = MyService()
    #     return _instance

The decorator also adds ``.reset()`` and ``.close()`` on the wrapper:

    get_my_service.reset()   # force next call to re-create
    get_my_service.close()   # calls instance.close() if it exists, then resets
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")

Factory = Callable[[], T]


def singleton(factory: Factory[T]) -> Factory[T] & Any:
    """Decorate a zero-arg factory to cache and reuse its first result.

    Thread-safe only if the factory itself is thread-safe (e.g. the inner
    constructor handles its own locking).  For thread-safe construction use
    a lock inside the factory function.
    """
    _instance: T | None = None
    _created: bool = False

    @wraps(factory)
    def wrapper() -> T:
        nonlocal _instance, _created
        if not _created:
            _instance = factory()
            _created = True
        assert _instance is not None  # factory must not return None
        return _instance

    def _reset() -> None:
        nonlocal _instance, _created
        _instance = None
        _created = False

    def _close() -> None:
        nonlocal _instance, _created
        if _instance is not None and hasattr(_instance, "close"):
            _instance.close()  # type: ignore[union-attr]
        _instance = None
        _created = False

    wrapper.reset = _reset   # type: ignore[attr-defined]
    wrapper.close = _close   # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
