"""Periodically polls Marketaux for a watchlist and persists articles.

Unlike QuotePoller, this issues ONE request per cycle covering all symbols
(Marketaux accepts a comma-separated symbol list), not one request per
symbol -- with a 100/day free-tier budget, one-request-per-symbol-per-cycle
would exhaust the entire daily quota in a handful of cycles.

A MarketauxQuotaExceededError stops the poller rather than retrying or
looping silently -- once the daily quota is gone, continuing to run just
produces a wall of identical failures until the quota resets.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg
import structlog

from marketpulse.ingestion.marketaux_client import MarketauxClient, MarketauxQuotaExceededError
from marketpulse.storage.news_repository import NewsRepository

logger = structlog.get_logger(__name__)


class NewsPoller:
    def __init__(
        self,
        client: MarketauxClient,
        pool: asyncpg.Pool,
        symbols: list[str],
        interval_seconds: float = 1800.0,  # 30 minutes by default
        articles_per_cycle: int = 10,
    ) -> None:
        if not symbols:
            raise ValueError("symbols list must not be empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        self._client = client
        self._repo = NewsRepository(pool)
        self._symbols = symbols
        self._interval = interval_seconds
        self._limit = articles_per_cycle
        self._stop_event = asyncio.Event()
        self._quota_exhausted = False

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def quota_exhausted(self) -> bool:
        return self._quota_exhausted

    async def _poll_once(self) -> None:
        try:
            response = await self._client.get_news_for_symbols(
                self._symbols, limit=self._limit
            )
        except MarketauxQuotaExceededError:
            logger.error("news_poll_quota_exhausted_stopping")
            self._quota_exhausted = True
            self.request_stop()
            return

        for article in response.articles:
            await self._repo.save(article)

        logger.info(
            "news_poll_saved",
            article_count=len(response.articles),
            symbols=self._symbols,
        )

    async def run(self) -> None:
        logger.info(
            "news_poller_starting",
            symbols=self._symbols,
            interval_seconds=self._interval,
        )
        while not self._stop_event.is_set():
            cycle_start = asyncio.get_event_loop().time()
            await self._poll_once()
            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_for = max(0.0, self._interval - elapsed)

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)

        logger.info("news_poller_stopped", quota_exhausted=self._quota_exhausted)
