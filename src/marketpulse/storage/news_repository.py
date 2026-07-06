"""Persistence layer for NewsArticle data.

An article and its entities are always written together in one transaction
-- an article with no entity rows, or entity rows pointing at a
never-inserted article, would both be silently-broken states that are
worse than just failing the whole write.
"""

from __future__ import annotations

import asyncpg

from marketpulse.ingestion.marketaux_models import NewsArticle, NewsEntity


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

