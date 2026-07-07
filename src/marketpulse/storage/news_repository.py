"""Persistence layer for NewsArticle data.

An article and its entities are always written together in one transaction
-- an article with no entity rows, or entity rows pointing at a
never-inserted article, would both be silently-broken states that are
worse than just failing the whole write.
"""

from __future__ import annotations

from datetime import UTC, datetime

import asyncpg

from marketpulse.ingestion.marketaux_models import NewsArticle, NewsEntity
from marketpulse.models.sentiment import SentimentLabel, SentimentScore


class NewsRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save(self, article: NewsArticle) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                INSERT INTO news_articles (uuid, title, description, url, source, published_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (uuid) DO NOTHING
                """,
                article.uuid,
                article.title,
                article.description,
                article.url,
                article.source,
                article.published_at,
            )
            for entity in article.entities:
                await conn.execute(
                    """
                    INSERT INTO article_entities
                        (article_uuid, symbol, name, sentiment_score, match_score)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (article_uuid, symbol) DO NOTHING
                    """,
                    article.uuid,
                    entity.symbol,
                    entity.name,
                    entity.sentiment_score,
                    entity.match_score,
                )

    async def save_many(self, articles: list[NewsArticle]) -> None:
        for article in articles:
            await self.save(article)

    async def recent_for_symbol(self, symbol: str, limit: int = 20) -> list[NewsArticle]:
        rows = await self._pool.fetch(
            """
            SELECT
                a.uuid, a.title, a.description, a.url, a.source, a.published_at,
                e.symbol AS entity_symbol, e.name AS entity_name,
                e.sentiment_score, e.match_score
            FROM news_articles a
            JOIN article_entities e ON e.article_uuid = a.uuid
            WHERE a.uuid IN (
                SELECT article_uuid FROM article_entities WHERE symbol = $1
            )
            ORDER BY a.published_at DESC
            LIMIT $2
            """,
            symbol,
            limit,
        )
        return self._rows_to_articles(rows)

    async def count(self) -> int:
        result = await self._pool.fetchval("SELECT count(*) FROM news_articles")
        return int(result)

    async def unscored_articles(self, limit: int = 50) -> list[NewsArticle]:
        """Fetch articles that haven't had sentiment scored yet.

        Returns bare articles (title/description populated, entities empty)
        -- entities aren't needed for scoring text, and fetching them here
        would mean an extra join for data the caller won't use.
        """
        rows = await self._pool.fetch(
            """
            SELECT uuid, title, description, url, source, published_at
            FROM news_articles
            WHERE sentiment_scored_at IS NULL
            ORDER BY published_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [
            NewsArticle(
                uuid=row["uuid"],
                title=row["title"],
                description=row["description"],
                url=row["url"],
                source=row["source"],
                published_at=row["published_at"],
                entities=[],
            )
            for row in rows
        ]

    async def save_sentiment(self, article_uuid: str, score: SentimentScore) -> None:
        await self._pool.execute(
            """
            UPDATE news_articles
            SET sentiment_label = $2,
                sentiment_confidence = $3,
                positive_prob = $4,
                negative_prob = $5,
                neutral_prob = $6,
                sentiment_scored_at = $7
            WHERE uuid = $1
            """,
            article_uuid,
            score.label.value,
            score.confidence,
            score.positive_prob,
            score.negative_prob,
            score.neutral_prob,
            datetime.now(UTC),
        )

    async def sentiment_for_symbol(
        self, symbol: str, limit: int = 20
    ) -> list[tuple[NewsArticle, SentimentLabel | None]]:
        """Recent articles for a symbol, paired with their sentiment label
        (None if not yet scored)."""
        rows = await self._pool.fetch(
            """
            SELECT DISTINCT ON (a.uuid)
                a.uuid, a.title, a.description, a.url, a.source, a.published_at,
                a.sentiment_label
            FROM news_articles a
            JOIN article_entities e ON e.article_uuid = a.uuid
            WHERE e.symbol = $1
            ORDER BY a.uuid, a.published_at DESC
            LIMIT $2
            """,
            symbol,
            limit,
        )
        return [
            (
                NewsArticle(
                    uuid=row["uuid"],
                    title=row["title"],
                    description=row["description"],
                    url=row["url"],
                    source=row["source"],
                    published_at=row["published_at"],
                    entities=[],
                ),
                SentimentLabel(row["sentiment_label"]) if row["sentiment_label"] else None,
            )
            for row in rows
        ]
    @staticmethod
    def _rows_to_articles(rows: list[asyncpg.Record]) -> list[NewsArticle]:
        # Rows are joined (one row per article-entity pair), so group back
        # into one NewsArticle per uuid with its full entity list attached.
        articles_by_uuid: dict[str, NewsArticle] = {}
        for row in rows:
            uuid = row["uuid"]
            if uuid not in articles_by_uuid:
                articles_by_uuid[uuid] = NewsArticle(
                    uuid=uuid,
                    title=row["title"],
                    description=row["description"],
                    url=row["url"],
                    source=row["source"],
                    published_at=row["published_at"],
                    entities=[],
                )
            articles_by_uuid[uuid].entities.append(
                NewsEntity(
                    symbol=row["entity_symbol"],
                    name=row["entity_name"],
                    sentiment_score=row["sentiment_score"],
                    match_score=row["match_score"],
                )
            )
        # Preserve published_at DESC ordering from the query.
        return list(articles_by_uuid.values())

