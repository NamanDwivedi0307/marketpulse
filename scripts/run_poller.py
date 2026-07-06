"""Entrypoint for continuous quote polling.

Usage:
    uv run python scripts/run_poller.py AAPL MSFT GOOGL --interval 60
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.finnhub_client import FinnhubClient
from marketpulse.ingestion.quote_poller import QuotePoller, install_signal_handlers
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.pool import create_pool
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll Finnhub quotes into TimescaleDB")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols to poll, e.g. AAPL MSFT")
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between poll cycles (default: 60)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    pool = await create_pool(settings.database)
    await run_migrations(pool)

    async with FinnhubClient(settings=settings) as client:
        poller = QuotePoller(
            client=client,
            pool=pool,
            symbols=[s.upper() for s in args.symbols],
            interval_seconds=args.interval,
        )
        loop = asyncio.get_running_loop()
        install_signal_handlers(poller, loop)

        await poller.run()

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
