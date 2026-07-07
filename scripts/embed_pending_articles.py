"""Generates embeddings for all news articles that don't have one yet.

Same pattern as score_pending_sentiment.py -- run periodically to keep the
embedding backlog from growing as new articles are ingested.

Usage:
    uv run python scripts/embed_pending_articles.py --batch-size 50
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.models.embeddings import EmbeddingService
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.pool import create_pool
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed pending news articles")
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Max articles to embed per run"
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging()
    settings = get_settings()

    pool = await create_pool(settings.database)
    await run_migrations(pool)
    repo = NewsRepository(pool)

    try:
        pending = await repo.articles_missing_embeddings(limit=args.batch_size)
        if not pending:
            logger.info("no_pending_articles_to_embed")
            return

        logger.info("embedding_starting", article_count=len(pending))

        embedder = EmbeddingService()

        texts = [f"{a.title}. {a.description}" for a in pending]
        vectors = embedder.embed_many(texts)

        for article, vector in zip(pending, vectors, strict=True):
            await repo.save_embedding(article.uuid, vector)
            logger.info("article_embedded", uuid=article.uuid, title=article.title[:60])

        logger.info("embedding_complete", article_count=len(pending))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
