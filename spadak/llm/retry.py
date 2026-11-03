"""Resilience helpers: exponential-backoff retries + token-bucket rate limiter."""
from __future__ import annotations

import random
import threading
import time
from functools import wraps
from typing import Any, Callable


RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def with_retries(max_attempts: int = 3, base_delay: float = 0.5,
                 max_delay: float = 20.0) -> Callable:
    """Decorator: retry on transient errors (incl. HTTPError status codes)."""
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    status = getattr(exc, "code", None) or _status_from_urllib(exc)
                    transient = status in RETRYABLE_STATUS if status else _looks_transient(exc)
                    if not transient or attempt == max_attempts - 1:
                        raise
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    time.sleep(delay + random.uniform(0, delay * 0.25))
            raise last_exc  # pragma: no cover
        return wrapper
    return deco


def _status_from_urllib(exc: Exception) -> int | None:
    import urllib.error
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    return None


def _looks_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in {"ConnectionError", "TimeoutError", "URLError",
                    "ReadTimeout", "RemoteDisconnected", "TemporarilyUnavailable"}


class RateLimiter:
    """Thread-safe token bucket: `acquire()` blocks until a slot frees up."""

    def __init__(self, requests_per_second: float = 2.0, burst: int | None = None) -> None:
        self.rate = requests_per_second
        self.capacity = burst if burst is not None else max(1, int(requests_per_second))
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity,
                                   self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
            time.sleep(0.01)
