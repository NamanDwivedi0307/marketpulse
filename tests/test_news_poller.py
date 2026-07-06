"""Unit tests for NewsPoller.

Mocks MarketauxClient and the repository's underlying pool -- orchestration
logic (single-call-per-cycle batching, quota-exhaustion stop behavior) is
what's under test here, not the real API or real Postgres.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from marketpulse.ingestion.marketaux_client import MarketauxQuotaExceededError
from marketpulse.ingestion.marketaux_models import NewsArticle, NewsResponse
from marketpulse.ingestion.news_poller import NewsPoller


def _article(uuid: str) -> NewsArticle:
    return NewsArticle(
        uuid=uuid,
        title=f"Article {uuid}",
        description="desc",
        url=f"https://example.com/{uuid}",
        source="test-source",
        published_at=datetime.now(UTC),
        entities=[],
    )


def test_rejects_empty_symbol_list() -> None:
    with pytest.raises(ValueError, match="symbols list"):
        NewsPoller(client=MagicMock(), pool=MagicMock(), symbols=[])


def test_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        NewsPoller(client=MagicMock(), pool=MagicMock(), symbols=["AAPL"], interval_seconds=0)


async def test_poll_once_issues_single_call_for_all_symbols() -> None:
    client = AsyncMock()
    client.get_news_for_symbols.return_value = NewsResponse(
        articles=[_article("a1"), _article("a2")]
    )
    # Bypass the real asyncpg pool entirely -- what's under test here is
    # NewsPoller's batching behavior, not NewsRepository's transaction
    # handling (already covered by test_news_repository_integration.py).
    poller = NewsPoller(client=client, pool=MagicMock(), symbols=["AAPL", "MSFT", "GOOGL"])
    poller._repo = AsyncMock()

    await poller._poll_once()

    client.get_news_for_symbols.assert_called_once()
    called_symbols = client.get_news_for_symbols.call_args[0][0]
    assert set(called_symbols) == {"AAPL", "MSFT", "GOOGL"}
    assert poller._repo.save.call_count == 2

async def test_quota_exceeded_sets_flag_and_requests_stop() -> None:
    client = AsyncMock()
    client.get_news_for_symbols.side_effect = MarketauxQuotaExceededError("quota gone")
    pool = MagicMock()

    poller = NewsPoller(client=client, pool=pool, symbols=["AAPL"])
    await poller._poll_once()

    assert poller.quota_exhausted is True


async def test_run_stops_promptly_on_request_stop() -> None:
    client = AsyncMock()
    client.get_news_for_symbols.return_value = NewsResponse(articles=[])
    pool = MagicMock()

    poller = NewsPoller(client=client, pool=pool, symbols=["AAPL"], interval_seconds=30.0)

    async def stop_after_first_cycle() -> None:
        await asyncio.sleep(0.05)
        poller.request_stop()

    await asyncio.wait_for(
        asyncio.gather(poller.run(), stop_after_first_cycle()),
        timeout=2.0,
    )
    assert client.get_news_for_symbols.call_count >= 1


async def test_run_stops_automatically_when_quota_exhausted() -> None:
    client = AsyncMock()
    client.get_news_for_symbols.side_effect = MarketauxQuotaExceededError("quota gone")
    pool = MagicMock()

    poller = NewsPoller(client=client, pool=pool, symbols=["AAPL"], interval_seconds=30.0)

    await asyncio.wait_for(poller.run(), timeout=2.0)  # should stop on its own, not hang
    assert poller.quota_exhausted is True
