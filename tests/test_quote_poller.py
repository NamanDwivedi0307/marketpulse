"""Unit tests for QuotePoller.

Mocks the FinnhubClient and a fake pool -- polling *orchestration* logic
(concurrency, error isolation, stop signaling) is what's under test here,
not the real Finnhub API or real Postgres, which are already covered
elsewhere (test_finnhub_client.py, test_storage_integration.py).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from marketpulse.ingestion.finnhub_client import FinnhubError
from marketpulse.ingestion.finnhub_models import Quote
from marketpulse.ingestion.quote_poller import QuotePoller


def _quote(symbol: str, price: float = 100.0) -> Quote:
    return Quote(
        symbol=symbol, c=price, d=1.0, dp=0.5,
        h=price + 1, l=price - 1, o=price, pc=price - 0.5,
        t=datetime.now(UTC),
    )


def test_rejects_empty_symbol_list() -> None:
    with pytest.raises(ValueError, match="symbols list"):
        QuotePoller(client=MagicMock(), pool=MagicMock(), symbols=[])


def test_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        QuotePoller(client=MagicMock(), pool=MagicMock(), symbols=["AAPL"], interval_seconds=0)


async def test_poll_all_once_fetches_and_saves_every_symbol() -> None:
    client = AsyncMock()
    client.get_quote.side_effect = lambda symbol: _quote(symbol)
    pool = MagicMock()
    pool.execute = AsyncMock()

    poller = QuotePoller(client=client, pool=pool, symbols=["AAPL", "MSFT"])
    await poller._poll_all_once()

    assert client.get_quote.call_count == 2
    assert pool.execute.call_count == 2


async def test_one_symbol_failure_does_not_block_others() -> None:
    client = AsyncMock()

    async def flaky_get_quote(symbol: str) -> Quote:
        if symbol == "BAD":
            raise FinnhubError("simulated failure")
        return _quote(symbol)

    client.get_quote.side_effect = flaky_get_quote
    pool = MagicMock()
    pool.execute = AsyncMock()

    poller = QuotePoller(client=client, pool=pool, symbols=["AAPL", "BAD", "MSFT"])
    await poller._poll_all_once()

    # AAPL and MSFT should have been saved despite BAD failing.
    assert pool.execute.call_count == 2


async def test_duplicate_quote_is_swallowed_not_raised() -> None:
    client = AsyncMock()
    client.get_quote.return_value = _quote("AAPL")
    pool = MagicMock()
    pool.execute = AsyncMock(side_effect=asyncpg.UniqueViolationError("duplicate"))

    poller = QuotePoller(client=client, pool=pool, symbols=["AAPL"])
    await poller._poll_all_once()  # should not raise


async def test_run_stops_promptly_on_request_stop() -> None:
    client = AsyncMock()
    client.get_quote.return_value = _quote("AAPL")
    pool = MagicMock()
    pool.execute = AsyncMock()

    poller = QuotePoller(client=client, pool=pool, symbols=["AAPL"], interval_seconds=30.0)

    async def stop_after_first_cycle() -> None:
        await asyncio.sleep(0.05)
        poller.request_stop()

    await asyncio.wait_for(
        asyncio.gather(poller.run(), stop_after_first_cycle()),
        timeout=2.0,
    )
    assert client.get_quote.call_count >= 1
