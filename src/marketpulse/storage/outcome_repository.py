"""Persistence layer for computed forward-return outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import asyncpg


@dataclass(frozen=True)
class EventOutcome:
    article_uuid: str
    symbol: str
    horizon_minutes: int
    entry_price: float
    entry_quoted_at: datetime
    exit_price: float
    exit_quoted_at: datetime
    return_pct: float


class OutcomeRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, outcome: EventOutcome) -> None:
        await self._pool.execute(
            """
            INSERT INTO event_outcomes (
                article_uuid, symbol, horizon_minutes,
                entry_price, entry_quoted_at, exit_price, exit_quoted_at, return_pct
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (article_uuid, symbol, horizon_minutes) DO NOTHING
            """,
            outcome.article_uuid,
            outcome.symbol,
            outcome.horizon_minutes,
            outcome.entry_price,
            outcome.entry_quoted_at,
            outcome.exit_price,
            outcome.exit_quoted_at,
            outcome.return_pct,
        )

    async def articles_missing_outcome(
        self, symbol: str, horizon_minutes: int, limit: int = 50
    ) -> list[tuple[str, datetime]]:
        """(article_uuid, published_at) pairs for a symbol/horizon combo
        that don't have an outcome computed yet."""
        rows = await self._pool.fetch(
            """
            SELECT a.uuid, a.published_at
            FROM news_articles a
            JOIN article_entities e ON e.article_uuid = a.uuid
            WHERE e.symbol = $1
              AND NOT EXISTS (
                  SELECT 1 FROM event_outcomes o
                  WHERE o.article_uuid = a.uuid
                    AND o.symbol = $1
                    AND o.horizon_minutes = $2
              )
            ORDER BY a.published_at DESC
            LIMIT $3
            """,
            symbol,
            horizon_minutes,
            limit,
        )
        return [(row["uuid"], row["published_at"]) for row in rows]

    async def average_return_for_similar(
        self, article_uuids: list[str], symbol: str, horizon_minutes: int
    ) -> float | None:
        """Average return_pct across a given set of article uuids (e.g. the
        top-k historically similar articles) for one symbol/horizon.

        Returns None if none of the given articles have a computed outcome
        for this symbol/horizon -- callers must treat that as "no historical
        precedent available," not as a return of 0%.
        """
        result = await self._pool.fetchval(
            """
            SELECT avg(return_pct)
            FROM event_outcomes
            WHERE article_uuid = ANY($1::text[])
              AND symbol = $2
              AND horizon_minutes = $3
            """,
            article_uuids,
            symbol,
            horizon_minutes,
        )
        return float(result) if result is not None else None
