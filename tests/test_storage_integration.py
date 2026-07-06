"""Integration tests against a real TimescaleDB instance.

Not mocked -- runs real migrations and real inserts/reads against whatever
Postgres the DATABASE_* settings point to. Skips cleanly if no database is
reachable, so the rest of the suite still runs fine without one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg
import pytest

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.finnhub_models import Quote
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.pool import create_pool
from marketpulse.storage.quote_repository import QuoteRepository


async def _db_reachable() -> bool:
    settings = get_settings()
    try:
        conn = await asyncpg.connect(dsn=settings.database.dsn, timeout=2.0)
    except (OSError, asyncpg.PostgresError):
        return False
    await conn.close()
    return True


@pytest.fixture
async def pool():
    settings = get_settings()
    if not await _db_reachable():
        pytest.skip(
            "No reachable Postgres/TimescaleDB at the configured DATABASE_* "
            "settings -- run `docker compose up -d` and retry."
        )
    p = await create_pool(settings.database)
    await run_migrations(p)
    yield p
    await p.execute("DELETE FROM quotes WHERE symbol LIKE 'TEST_SYM%'")
    await p.close()


def _make_quote(symbol: str, price: float, quoted_at: datetime) -> Quote:
    return Quote(
        symbol=symbol,
        c=price,
        d=1.0,
        dp=0.5,
        h=price + 1,
        l=price - 1,
        o=price,
        pc=price - 0.5,
        t=quoted_at,
    )


async def test_migrations_apply_cleanly(pool: asyncpg.Pool) -> None:
    tables = await pool.fetch(
        "SELECT tablename FROM pg_tables WHERE tablename = 'quotes'"
    )
    assert len(tables) == 1


async def test_save_and_retrieve_latest_quote(pool: asyncpg.Pool) -> None:
    repo = QuoteRepository(pool)
    quote = _make_quote("TEST_SYM1", 150.0, datetime.now(UTC))

    await repo.save(quote)
    stored = await repo.latest_for_symbol("TEST_SYM1")

    assert stored is not None
    assert stored.symbol == "TEST_SYM1"
    assert stored.current_price == pytest.approx(150.0)


async def test_latest_for_symbol_returns_none_when_absent(pool: asyncpg.Pool) -> None:
    repo = QuoteRepository(pool)
    result = await repo.latest_for_symbol("TEST_SYM_NEVER_INSERTED")
    assert result is None


async def test_save_many_persists_all_rows(pool: asyncpg.Pool) -> None:
    repo = QuoteRepository(pool)
    now = datetime.now(UTC)
    quotes = [
        _make_quote("TEST_SYM2", 100.0 + i, now.replace(microsecond=i * 1000))
        for i in range(5)
    ]

    await repo.save_many(quotes)
    count = await repo.count_for_symbol("TEST_SYM2")

    assert count == 5


async def test_latest_for_symbol_returns_most_recent_not_first_inserted(
    pool: asyncpg.Pool,
) -> None:
    repo = QuoteRepository(pool)
    earlier = datetime(2024, 1, 1, tzinfo=UTC)
    later = datetime(2024, 6, 1, tzinfo=UTC)

    await repo.save(_make_quote("TEST_SYM3", 100.0, earlier))
    await repo.save(_make_quote("TEST_SYM3", 200.0, later))

    stored = await repo.latest_for_symbol("TEST_SYM3")
    assert stored is not None
    assert stored.current_price == pytest.approx(200.0)
