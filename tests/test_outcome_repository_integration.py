"""Integration tests for OutcomeRepository and the forward-return logic,
using synthetic quotes/articles with controlled timestamps -- this proves
the entry/exit price lookup and return calculation are correct without
needing to wait for real market hours.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import asyncpg
import pytest

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.finnhub_models import Quote
from marketpulse.ingestion.marketaux_models import NewsArticle, NewsEntity
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.outcome_repository import EventOutcome, OutcomeRepository
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
            "No reachable Postgres/TimescaleDB -- run `docker compose up -d` and retry."
        )
    p = await create_pool(settings.database)
    await run_migrations(p)
    yield p
    await p.execute("DELETE FROM event_outcomes WHERE symbol LIKE 'TEST_SYM%'")
    await p.execute("DELETE FROM quotes WHERE symbol LIKE 'TEST_SYM%'")
    await p.execute("DELETE FROM news_articles WHERE uuid LIKE 'TEST_OUTCOME%'")
    await p.close()


def _quote(symbol: str, price: float, quoted_at: datetime) -> Quote:
    return Quote(
        symbol=symbol, c=price, d=1.0, dp=0.5,
        h=price + 1, l=price - 1, o=price, pc=price - 0.5,
        t=quoted_at,
    )


def _article(uuid: str, symbol: str, published_at: datetime) -> NewsArticle:
    return NewsArticle(
        uuid=uuid,
        title=f"Synthetic article {uuid}",
        description="Test description",
        url=f"https://example.com/{uuid}",
        source="test-source",
        published_at=published_at,
        entities=[NewsEntity(symbol=symbol, name="Test Corp")],
    )


async def test_quote_at_or_after_finds_earliest_matching_quote(pool: asyncpg.Pool) -> None:
    repo = QuoteRepository(pool)
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)

    await repo.save(_quote("TEST_SYM1", 100.0, base))
    await repo.save(_quote("TEST_SYM1", 105.0, base + timedelta(minutes=5)))
    await repo.save(_quote("TEST_SYM1", 110.0, base + timedelta(minutes=10)))

    # Query for a time between the first and second quote -- should return
    # the second one (earliest AT OR AFTER), not the first or third.
    result = await repo.quote_at_or_after("TEST_SYM1", base + timedelta(minutes=2))

    assert result is not None
    assert result.current_price == pytest.approx(105.0)


async def test_quote_at_or_after_returns_none_when_no_future_quote_exists(
    pool: asyncpg.Pool,
) -> None:
    repo = QuoteRepository(pool)
    base = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    await repo.save(_quote("TEST_SYM2", 100.0, base))

    # Asking for a time after the only quote we have -- correctly finds nothing.
    result = await repo.quote_at_or_after("TEST_SYM2", base + timedelta(hours=1))
    assert result is None


async def test_end_to_end_outcome_computation_with_synthetic_data(pool: asyncpg.Pool) -> None:
    """Mirrors exactly what compute_event_outcomes.py does, but with fully
    controlled synthetic timestamps so it doesn't depend on real market hours.
    """
    quote_repo = QuoteRepository(pool)
    news_repo = NewsRepository(pool)
    outcome_repo = OutcomeRepository(pool)

    article_time = datetime(2024, 3, 1, 14, 0, 0, tzinfo=UTC)
    horizon = 30  # minutes

    article = _article("TEST_OUTCOME_1", "TEST_SYM3", article_time)
    await news_repo.save(article)

    # Entry quote: right at article publish time.
    await quote_repo.save(_quote("TEST_SYM3", 100.0, article_time))
    # Exit quote: 30 minutes later, price up 5%.
    await quote_repo.save(_quote("TEST_SYM3", 105.0, article_time + timedelta(minutes=horizon)))

    pending = await outcome_repo.articles_missing_outcome("TEST_SYM3", horizon)
    assert len(pending) == 1
    assert pending[0][0] == "TEST_OUTCOME_1"

    entry = await quote_repo.quote_at_or_after("TEST_SYM3", article_time)
    exit_quote = await quote_repo.quote_at_or_after(
        "TEST_SYM3", article_time + timedelta(minutes=horizon)
    )
    assert entry is not None
    assert exit_quote is not None

    return_pct = ((exit_quote.current_price - entry.current_price) / entry.current_price) * 100
    assert return_pct == pytest.approx(5.0)

    await outcome_repo.save(
        EventOutcome(
            article_uuid="TEST_OUTCOME_1",
            symbol="TEST_SYM3",
            horizon_minutes=horizon,
            entry_price=entry.current_price,
            entry_quoted_at=entry.quoted_at,
            exit_price=exit_quote.current_price,
            exit_quoted_at=exit_quote.quoted_at,
            return_pct=return_pct,
        )
    )

    # Now that an outcome exists, it should no longer show up as pending.
    pending_after = await outcome_repo.articles_missing_outcome("TEST_SYM3", horizon)
    assert pending_after == []


async def test_average_return_for_similar_computes_correct_mean(pool: asyncpg.Pool) -> None:
    outcome_repo = OutcomeRepository(pool)
    news_repo = NewsRepository(pool)
    quote_repo = QuoteRepository(pool)

    base = datetime(2024, 4, 1, 9, 0, 0, tzinfo=UTC)
    horizon = 15

    # Two synthetic articles/outcomes: +4% and +8%, average should be +6%.
    for i, pct_move in enumerate([4.0, 8.0]):
        uuid = f"TEST_OUTCOME_AVG_{i}"
        article_time = base + timedelta(hours=i)
        await news_repo.save(_article(uuid, "TEST_SYM4", article_time))

        entry_price = 100.0
        exit_price = entry_price * (1 + pct_move / 100)
        await quote_repo.save(_quote("TEST_SYM4", entry_price, article_time))
        await quote_repo.save(
            _quote("TEST_SYM4", exit_price, article_time + timedelta(minutes=horizon))
        )

        await outcome_repo.save(
            EventOutcome(
                article_uuid=uuid,
                symbol="TEST_SYM4",
                horizon_minutes=horizon,
                entry_price=entry_price,
                entry_quoted_at=article_time,
                exit_price=exit_price,
                exit_quoted_at=article_time + timedelta(minutes=horizon),
                return_pct=pct_move,
            )
        )

    avg = await outcome_repo.average_return_for_similar(
        ["TEST_OUTCOME_AVG_0", "TEST_OUTCOME_AVG_1"], "TEST_SYM4", horizon
    )
    assert avg == pytest.approx(6.0)


async def test_average_return_returns_none_when_no_outcomes_exist(pool: asyncpg.Pool) -> None:
    outcome_repo = OutcomeRepository(pool)
    avg = await outcome_repo.average_return_for_similar(
        ["TEST_OUTCOME_NEVER_EXISTED"], "TEST_SYM5", 60
    )
    assert avg is None
