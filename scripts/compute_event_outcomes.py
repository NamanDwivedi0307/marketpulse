"""Computes forward-return outcomes for news articles against a watchlist.

For each article mentioning a watched symbol, finds the quote at (or just
after) the article's publish time as the entry price, and the quote at
publish_time + horizon as the exit price, then stores the percentage return.

Requires the quote poller to have been running long enough that quotes
actually exist spanning an article's publish time through publish_time +
horizon -- an article published before polling started, or one too recent
for the horizon to have elapsed yet, will simply have no outcome computed
for it (skipped, not treated as an error).

Usage:
    uv run python scripts/compute_event_outcomes.py AAPL MSFT --horizon 60
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.outcome_repository import EventOutcome, OutcomeRepository
from marketpulse.storage.pool import create_pool
from marketpulse.storage.quote_repository import QuoteRepository
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute forward-return event outcomes")
    parser.add_argument("symbols", nargs="+", help="Symbols to compute outcomes for")
    parser.add_argument(
        "--horizon", type=int, default=60, help="Forward-return horizon in minutes (default: 60)"
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    pool = await create_pool(settings.database)
    await run_migrations(pool)

    quote_repo = QuoteRepository(pool)
    outcome_repo = OutcomeRepository(pool)

    try:
        for symbol in (s.upper() for s in args.symbols):
            pending = await outcome_repo.articles_missing_outcome(
                symbol, args.horizon, limit=100
            )
            if not pending:
                logger.info("no_pending_outcomes", symbol=symbol)
                continue

            computed = 0
            skipped = 0
            for article_uuid, published_at in pending:
                entry = await quote_repo.quote_at_or_after(symbol, published_at)
                if entry is None:
                    skipped += 1
                    continue

                exit_time = published_at + timedelta(minutes=args.horizon)
                exit_quote = await quote_repo.quote_at_or_after(symbol, exit_time)
                if exit_quote is None:
                    skipped += 1
                    continue

                return_pct = (
                    (exit_quote.current_price - entry.current_price) / entry.current_price
                ) * 100

                await outcome_repo.save(
                    EventOutcome(
                        article_uuid=article_uuid,
                        symbol=symbol,
                        horizon_minutes=args.horizon,
                        entry_price=entry.current_price,
                        entry_quoted_at=entry.quoted_at,
                        exit_price=exit_quote.current_price,
                        exit_quoted_at=exit_quote.quoted_at,
                        return_pct=return_pct,
                    )
                )
                computed += 1
                logger.info(
                    "outcome_computed",
                    symbol=symbol,
                    article_uuid=article_uuid,
                    return_pct=round(return_pct, 3),
                )

            logger.info(
                "symbol_outcomes_done", symbol=symbol, computed=computed, skipped=skipped
            )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
