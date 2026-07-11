"""Backfills historical daily OHLC quotes from yfinance.

Finnhub's free tier blocks the /stock/candle endpoint (403), so there's no
way to get historical bars from our primary provider without a paid plan.
yfinance (unofficial Yahoo Finance wrapper, no API key required) fills that
gap for backfill purposes only -- the live quote_poller keeps using Finnhub
for real-time data, this script is a one-time/occasional bulk-load path to
give the forecasting model enough historical rows to train on.

Derives change/percent_change from consecutive closes since yfinance daily
bars don't carry Finnhub's intraday change fields directly.

Usage:
    uv run python scripts/backfill_historical_quotes.py AAPL MSFT GOOGL --period 2y
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

import structlog
import yfinance as yf

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.finnhub_models import Quote
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.pool import create_pool
from marketpulse.storage.quote_repository import QuoteRepository
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical daily quotes via yfinance")
    parser.add_argument("symbols", nargs="+", help="Symbols to backfill")
    parser.add_argument(
        "--period", default="2y", help="yfinance period string, e.g. '2y', '5y', 'max'"
    )
    return parser.parse_args()


def quotes_from_yfinance(symbol: str, period: str) -> list[Quote]:
    df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        logger.warning("no_yfinance_data", symbol=symbol)
        return []

    # yf.download on a single symbol still returns MultiIndex columns in
    # recent yfinance versions -- flatten to the plain field name.
    if isinstance(df.columns, type(df.columns)) and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    quotes: list[Quote] = []
    previous_close: float | None = None

    for date, row in df.iterrows():
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        open_ = float(row["Open"])

        if previous_close is None:
            previous_close = close
            continue

        change = close - previous_close
        percent_change = (change / previous_close) * 100 if previous_close else 0.0
        quoted_at = datetime.combine(date.date(), datetime.min.time(), tzinfo=UTC)

        try:
            quotes.append(
                Quote(
                    symbol=symbol,
                    c=close,
                    d=change,
                    dp=percent_change,
                    h=high,
                    l=low,
                    o=open_,
                    pc=previous_close,
                    t=quoted_at,
                )
            )
        except ValueError as exc:
            logger.warning("skipping_invalid_row", symbol=symbol, date=str(date), error=str(exc))

        previous_close = close

    return quotes


async def main() -> None:
    args = parse_args()
    configure_logging()
    settings = get_settings()
    pool = await create_pool(settings.database)
    await run_migrations(pool)
    quote_repo = QuoteRepository(pool)

    try:
        for symbol in (s.upper() for s in args.symbols):
            quotes = quotes_from_yfinance(symbol, args.period)
            if not quotes:
                continue
            await quote_repo.save_many(quotes)
            logger.info("backfill_saved", symbol=symbol, rows=len(quotes))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
