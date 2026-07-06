"""Runs pending migrations, then fetches a live quote and persists it.

Manual verification script -- proves the ingestion -> storage path works
end to end against a real database and a real API, in one command.

Usage:
    uv run python scripts/verify_storage_pipeline.py AAPL
"""

from __future__ import annotations

import asyncio
import sys

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.finnhub_client import FinnhubClient
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.pool import create_pool
from marketpulse.storage.quote_repository import QuoteRepository
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


async def main(symbol: str) -> None:
    configure_logging()
    settings = get_settings()

    pool = await create_pool(settings.database)
    try:
        applied = await run_migrations(pool)
        if applied:
            logger.info("migrations_applied", files=applied)
        else:
            logger.info("migrations_up_to_date")

        repo = QuoteRepository(pool)

        async with FinnhubClient(settings=settings) as client:
            quote = await client.get_quote(symbol)

        logger.info(
            "quote_fetched",
            symbol=quote.symbol,
            price=quote.current_price,
            quoted_at=quote.quoted_at.isoformat(),
        )

        await repo.save(quote)
        logger.info("quote_persisted", symbol=symbol)

        stored = await repo.latest_for_symbol(symbol)
        count = await repo.count_for_symbol(symbol)

        if stored is None:
            raise RuntimeError("Wrote a quote but could not read it back")

        logger.info(
            "quote_readback_verified",
            symbol=stored.symbol,
            price=stored.current_price,
            total_rows_for_symbol=count,
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(main(ticker))

