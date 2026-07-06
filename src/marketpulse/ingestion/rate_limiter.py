"""A token-bucket rate limiter for async API clients.

A naive rate limiter (sleep(60/max_calls) between every call) wastes most of
its budget on quiet periods and still bursts incorrectly under concurrent
callers. A token bucket lets calls through immediately as long as capacity is
available, refills continuously, and blocks only once the budget is actually
exhausted -- which is both faster in practice and correct under concurrency.
"""

from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    def __init__(self, max_tokens: int, refill_period_seconds: float) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if refill_period_seconds <= 0:
            raise ValueError("refill_period_seconds must be positive")

        self._max_tokens = max_tokens
        self._refill_period = refill_period_seconds
        self._tokens = float(max_tokens)
        self._refill_rate = max_tokens / refill_period_seconds
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                deficit = 1 - self._tokens
                wait_time = deficit / self._refill_rate
            # Sleep outside the lock so other waiters can check refill progress
            # independently rather than serializing behind this one sleep.
            await asyncio.sleep(wait_time)
