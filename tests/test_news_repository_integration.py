"""Integration tests for NewsRepository against a real TimescaleDB instance.

Same reachability-gated pattern as test_storage_integration.py -- skips
cleanly if no database is running.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg
import pytest

from marketpulse.config.settings import get_settings
from marketpulse.ingestion.marketaux_models import NewsArticle, NewsEntity
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.pool import create_pool


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
    await p.execute("DELETE FROM news_articles WHERE uuid LIKE 'TEST_UUID%'")
    await p.close()


def _make_article(
    uuid: str,
    symbol: str = "TEST_SYM",
    published_at: datetime | None = None,
) -> NewsArticle:
    return NewsArticle(
        uuid=uuid,
        title=f"Test article {uuid}",
        description="A test description",
        url=f"https://example.com/{uuid}",
        source="test-source",
        published_at=published_at or datetime.now(UTC),
        entities=[
            NewsEntity(symbol=symbol, name="Test Corp", sentiment_score=0.5, match_score=0.9)
        ],
    )


async def test_save_and_retrieve_article_by_symbol(pool: asyncpg.Pool) -> None:
    repo = NewsRepository(pool)
    article = _make_article("TEST_UUID1", symbol="TEST_SYM1")

    await repo.save(article)
    results = await repo.recent_for_symbol("TEST_SYM1")

    assert len(results) == 1
    assert results[0].uuid == "TEST_UUID1"
    assert results[0].entities[0].symbol == "TEST_SYM1"


async def test_recent_for_symbol_returns_empty_when_no_match(pool: asyncpg.Pool) -> None:
    repo = NewsRepository(pool)
    results = await repo.recent_for_symbol("TEST_SYM_NEVER_SEEN")
    assert results == []


async def test_save_is_idempotent_on_duplicate_uuid(pool: asyncpg.Pool) -> None:
    repo = NewsRepository(pool)
    article = _make_article("TEST_UUID2", symbol="TEST_SYM2")

    await repo.save(article)
    await repo.save(article)  # saving the same article twice must not error or duplicate

    results = await repo.recent_for_symbol("TEST_SYM2")
    assert len(results) == 1


async def test_save_many_persists_all_articles(pool: asyncpg.Pool) -> None:
    repo = NewsRepository(pool)
    articles = [_make_article(f"TEST_UUID_BATCH_{i}", symbol="TEST_SYM3") for i in range(3)]

    await repo.save_many(articles)
    results = await repo.recent_for_symbol("TEST_SYM3")

    assert len(results) == 3


async def test_recent_for_symbol_orders_most_recent_first(pool: asyncpg.Pool) -> None:
    repo = NewsRepository(pool)
    earlier = datetime(2024, 1, 1, tzinfo=UTC)
    later = datetime(2024, 6, 1, tzinfo=UTC)

    await repo.save(_make_article("TEST_UUID_EARLY", symbol="TEST_SYM4", published_at=earlier))
    await repo.save(_make_article("TEST_UUID_LATE", symbol="TEST_SYM4", published_at=later))

    results = await repo.recent_for_symbol("TEST_SYM4")
    assert results[0].uuid == "TEST_UUID_LATE"
