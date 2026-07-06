import asyncio
import time

import pytest

from marketpulse.ingestion.rate_limiter import TokenBucketRateLimiter


async def test_allows_burst_up_to_capacity() -> None:
    limiter = TokenBucketRateLimiter(max_tokens=5, refill_period_seconds=1.0)
    start = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.monotonic() - start
    # All 5 tokens should be available immediately, no waiting.
    assert elapsed < 0.05


async def test_blocks_once_capacity_exhausted() -> None:
    limiter = TokenBucketRateLimiter(max_tokens=2, refill_period_seconds=0.2)
    await limiter.acquire()
    await limiter.acquire()

    start = time.monotonic()
    await limiter.acquire()  # third call must wait for a refill
    elapsed = time.monotonic() - start

    assert elapsed >= 0.05  # some real wait occurred, not an instant pass-through


async def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        TokenBucketRateLimiter(max_tokens=0, refill_period_seconds=1.0)
    with pytest.raises(ValueError, match="refill_period_seconds"):
        TokenBucketRateLimiter(max_tokens=5, refill_period_seconds=0)


async def test_concurrent_callers_all_eventually_proceed() -> None:
    limiter = TokenBucketRateLimiter(max_tokens=3, refill_period_seconds=0.3)
    results = await asyncio.gather(*(limiter.acquire() for _ in range(6)))
    assert len(results) == 6

