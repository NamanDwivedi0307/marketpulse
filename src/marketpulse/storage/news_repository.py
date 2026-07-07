"""Persistence layer for NewsArticle data.

Deliberately not a generic "repository base class" -- with one entity type
so far, an abstraction over an abstraction just hides the actual SQL being
run. Add a shared base once there are three or four of these with real
duplication, not before.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import asyncpg

from marketpulse.ingestion.marketaux_models import NewsArticle, NewsEntity
from marketpulse.models.sentiment import SentimentLabel, SentimentScore


@dataclass(frozen=True)
class SimilarArticle:
    uuid: str
    title: str
    sentiment_label: SentimentLabel
    sentiment_confidence: float
    similarity: float


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

    async def save_embedding(self, article_uuid: str, embedding: list[float]) -> None:
        # pgvector expects the literal string format '[0.1,0.2,...]', not a
        # Python list -- asyncpg has no native vector type, so we format it
        # ourselves rather than depending on an extra pgvector-specific driver.
        vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        await self._pool.execute(
            "UPDATE news_articles SET embedding = $2 WHERE uuid = $1",
            article_uuid,
            vector_literal,
        )

    async def articles_missing_embeddings(self, limit: int = 50) -> list[NewsArticle]:
        rows = await self._pool.fetch(
            """
            SELECT uuid, title, description, url, source, published_at
            FROM news_articles
            WHERE embedding IS NULL
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

    async def most_similar_articles(
        self, embedding: list[float], limit: int = 5, exclude_uuid: str | None = None
    ) -> list[SimilarArticle]:
        """Find the most semantically similar scored articles to a given
        embedding, using cosine distance via pgvector's <=> operator.

        Only returns articles that already have a sentiment label -- an
        unscored match is useless to the historical-event engine, since the
        entire point is retrieving "what sentiment did similar past events
        carry."
        """
        vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
        rows = await self._pool.fetch(
            """
            SELECT uuid, title, sentiment_label, sentiment_confidence,
                   1 - (embedding <=> $1) AS similarity
            FROM news_articles
            WHERE embedding IS NOT NULL
              AND sentiment_label IS NOT NULL
              AND ($2::text IS NULL OR uuid != $2)
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            vector_literal,
            exclude_uuid,
            limit,
        )
        return [
            SimilarArticle(
                uuid=row["uuid"],
                title=row["title"],
                sentiment_label=SentimentLabel(row["sentiment_label"]),
                sentiment_confidence=row["sentiment_confidence"],
                similarity=row["similarity"],
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
        return list(articles_by_uuid.values())
