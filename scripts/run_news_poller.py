"""Entrypoint for continuous news polling.

Usage:
    uv run python scripts/run_news_poller.py AAPL MSFT GOOGL --interval 1800
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.marketaux_client import MarketauxClient
from marketpulse.ingestion.news_poller import NewsPoller
from marketpulse.ingestion.quote_poller import install_signal_handlers
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.pool import create_pool
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Marketaux news into TimescaleDB")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to poll, e.g. AAPL MSFT")
    parser.add_argument(
        "--interval",
        type=float,
        default=1800.0,
        help="Seconds between poll cycles (default: 1800 = 30 min, mind the 100/day free quota)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max articles fetched per cycle (default: 10)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    pool = await create_pool(settings.database)
    await run_migrations(pool)

    async with MarketauxClient(settings=settings) as client:
        poller = NewsPoller(
            client=client,
            pool=pool,
            symbols=[s.upper() for s in args.symbols],
            interval_seconds=args.interval,
            articles_per_cycle=args.limit,
        )
        loop = asyncio.get_running_loop()
        install_signal_handlers(poller, loop)

        await poller.run()

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
