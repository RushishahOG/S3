"""Small reusable utilities: retry decorator and stable hashing."""

from __future__ import annotations

import functools
import time
from typing import Callable

from core.utils.logging_config import get_logger

logger = get_logger(__name__)


def retry(
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    max_sleep_per_attempt: float | None = None,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """Decorator that retries a callable with linear backoff.

    Used primarily by data providers to gracefully handle transient network
    failures without coupling retry logic into provider implementations.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (initial call + retries). Hard-capped at 3
        to avoid runaway retry storms against external APIs.
    backoff_seconds : float
        Base linear backoff between attempts.
    max_sleep_per_attempt : float | None
        Upper bound for a single sleep interval. When set, the per-attempt delay
        is ``min(backoff_seconds * attempt, max_sleep_per_attempt)`` so retries
        stay bounded even with large base backoffs.
    """

    # Hard cap to prevent runaway retries against external APIs.
    max_attempts = min(int(max_attempts), 3)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = backoff_seconds * attempt
                    if max_sleep_per_attempt is not None:
                        delay = min(delay, max_sleep_per_attempt)
                    logger.warning(
                        "Function %s attempt %d/%d failed: %s. Retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            # Unreachable, but keeps type-checkers happy.
            raise RuntimeError("retry exhausted")  # pragma: no cover

        return wrapper

    return decorator
