"""Persistence layer for Quote data.

Deliberately not a generic "repository base class" -- with one entity type
so far, an abstraction over an abstraction just hides the actual SQL being
run. Add a shared base once there are three or four of these with real
duplication, not before.
"""

from __future__ import annotations

from datetime import datetime

import asyncpg

from marketpulse.ingestion.finnhub_models import Quote


class QuoteRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, quote: Quote) -> None:
        await self._pool.execute(
            """
            INSERT INTO quotes (
                symbol, current_price, change, percent_change,
                high_of_day, low_of_day, open_price, previous_close, quoted_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            quote.symbol,
            quote.current_price,
            quote.change,
            quote.percent_change,
            quote.high_of_day,
            quote.low_of_day,
            quote.open_price,
            quote.previous_close,
            quote.quoted_at,
        )

    async def save_many(self, quotes: list[Quote]) -> None:
        if not quotes:
            return
        await self._pool.executemany(
            """
            INSERT INTO quotes (
                symbol, current_price, change, percent_change,
                high_of_day, low_of_day, open_price, previous_close, quoted_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            [
                (
                    q.symbol,
                    q.current_price,
                    q.change,
                    q.percent_change,
                    q.high_of_day,
                    q.low_of_day,
                    q.open_price,
                    q.previous_close,
                    q.quoted_at,
                )
                for q in quotes
            ],
        )

    async def latest_for_symbol(self, symbol: str) -> Quote | None:
        row = await self._pool.fetchrow(
            """
            SELECT symbol, current_price, change, percent_change,
                   high_of_day, low_of_day, open_price, previous_close, quoted_at
            FROM quotes
            WHERE symbol = $1
            ORDER BY quoted_at DESC
            LIMIT 1
            """,
            symbol,
        )
        if row is None:
            return None
        return Quote(
            symbol=row["symbol"],
            c=row["current_price"],
            d=row["change"],
            dp=row["percent_change"],
            h=row["high_of_day"],
            l=row["low_of_day"],
            o=row["open_price"],
            pc=row["previous_close"],
            t=row["quoted_at"],
        )

    async def count_for_symbol(self, symbol: str) -> int:
        result = await self._pool.fetchval(
            "SELECT count(*) FROM quotes WHERE symbol = $1", symbol
        )
        return int(result)

    async def recent_for_symbol(self, symbol: str, limit: int = 30) -> list[Quote]:
        """Most recent `limit` quotes for a symbol, oldest first.

        Oldest-first ordering matters for callers computing rolling
        technical features (momentum, volatility) -- those need
        chronological order, not most-recent-first.
        """
        # DISTINCT ON (quoted_at::date) collapses same-day duplicate polls
        # down to one row per calendar day (the latest poll that day) --
        # without this, a symbol polled every 60s on a day when the
        # upstream price snapshot doesn't change yields dozens of identical
        # rows for one date, silently starving rolling-window feature
        # calculations of the distinct trading days they actually need.
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT ON (quoted_at::date)
                symbol, current_price, change, percent_change,
                high_of_day, low_of_day, open_price, previous_close, quoted_at
            FROM quotes
            WHERE symbol = $1
            ORDER BY quoted_at::date DESC, quoted_at DESC
            LIMIT $2
            """,
            symbol,
            limit,
        )
        quotes = [
            Quote(
                symbol=row["symbol"],
                c=row["current_price"],
                d=row["change"],
                dp=row["percent_change"],
                h=row["high_of_day"],
                l=row["low_of_day"],
                o=row["open_price"],
                pc=row["previous_close"],
                t=row["quoted_at"],
            )
            for row in rows
        ]
        return list(reversed(quotes))

    async def quote_at_or_after(self, symbol: str, timestamp: datetime) -> Quote | None:
        """The earliest quote for symbol at or after the given timestamp.

        Used for forward-return computation: given a news event at time T,
        this finds the actual price snapshot closest to T (not necessarily
        exactly T, since polling happens on a fixed interval, not on demand)
        to use as the entry price for a return calculation.
        """
        row = await self._pool.fetchrow(
            """
            SELECT symbol, current_price, change, percent_change,
                   high_of_day, low_of_day, open_price, previous_close, quoted_at
            FROM quotes
            WHERE symbol = $1 AND quoted_at >= $2
            ORDER BY quoted_at ASC
            LIMIT 1
            """,
            symbol,
            timestamp,
        )
        if row is None:
            return None
        return Quote(
            symbol=row["symbol"],
            c=row["current_price"],
            d=row["change"],
            dp=row["percent_change"],
            h=row["high_of_day"],
            l=row["low_of_day"],
            o=row["open_price"],
            pc=row["previous_close"],
            t=row["quoted_at"],
        )
