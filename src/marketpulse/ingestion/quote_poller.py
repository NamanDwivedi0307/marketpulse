"""Continuously polls Finnhub for a watchlist of symbols and persists quotes.

Runs as a long-lived asyncio loop rather than a cron job -- this keeps one
warm FinnhubClient (and its rate limiter state) alive across the whole
polling session instead of paying connection/auth overhead on every tick,
and makes it trivial to run many symbols concurrently within a single
rate-limit budget.

Failures on individual symbols are logged and skipped, not fatal -- one bad
symbol or one transient network blip should never take down polling for the
rest of the watchlist.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

import asyncpg
import structlog

from marketpulse.ingestion.finnhub_client import FinnhubClient, FinnhubError
from marketpulse.storage.quote_repository import QuoteRepository

logger = structlog.get_logger(__name__)


class QuotePoller:
    def __init__(
        self,
        client: FinnhubClient,
        pool: asyncpg.Pool,
        symbols: list[str],
        interval_seconds: float = 60.0,
    ) -> None:
        if not symbols:
            raise ValueError("symbols list must not be empty")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        self._client = client
        self._repo = QuoteRepository(pool)
        self._symbols = symbols
        self._interval = interval_seconds
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    async def _poll_one(self, symbol: str) -> None:
        log = logger.bind(symbol=symbol)
        try:
            quote = await self._client.get_quote(symbol)
        except FinnhubError as exc:
            log.warning("quote_poll_failed", error=str(exc))
            return

        try:
            await self._repo.save(quote)
        except asyncpg.UniqueViolationError:
            # Expected during illiquid periods: Finnhub can return the same
            # quoted_at on back-to-back polls if no new trade has occurred.
            log.debug("quote_poll_duplicate_skipped", quoted_at=quote.quoted_at.isoformat())
            return

        log.info("quote_poll_saved", price=quote.current_price)

    async def _poll_all_once(self) -> None:
        # Concurrent, not sequential -- the rate limiter inside FinnhubClient
        # already enforces the real ceiling, so there's no reason to poll
        # symbols one at a time and eat N * request_latency per cycle.
        await asyncio.gather(*(self._poll_one(s) for s in self._symbols))

    async def run(self) -> None:
        logger.info(
            "poller_starting",
            symbols=self._symbols,
            interval_seconds=self._interval,
        )
        while not self._stop_event.is_set():
            cycle_start = asyncio.get_event_loop().time()
            await self._poll_all_once()
            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_for = max(0.0, self._interval - elapsed)

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_for)

        logger.info("poller_stopped")


def install_signal_handlers(poller: QuotePoller, loop: asyncio.AbstractEventLoop) -> None:
    """Wire SIGINT/SIGTERM to a graceful stop instead of a hard kill.

    Without this, Ctrl+C during an in-flight poll cycle can leave a
    partially-written batch or an unclosed HTTP connection.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, poller.request_stop)
