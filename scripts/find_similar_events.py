"""Given a piece of breaking news text, finds the most similar historical
articles and reports their sentiment -- the core "historical event engine"
mechanism: embed the live event, retrieve nearest neighbors by cosine
similarity, and surface what sentiment those past events carried as a
statistical precedent for the new one.

This does not yet predict price movement (that requires the forward-return
labeling described in the original architecture doc); it answers the
prerequisite question -- "have we seen something like this before, and how
did it read?" -- which the price-prediction fusion layer will build on.

Usage:
    uv run python scripts/find_similar_events.py "Nvidia announces new AI chip architecture"
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.models.embeddings import EmbeddingService
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.pool import create_pool
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


async def main(query_text: str, top_k: int) -> None:
    configure_logging()
    settings = get_settings()

    pool = await create_pool(settings.database)
    await run_migrations(pool)
    repo = NewsRepository(pool)

    try:
        embedder = EmbeddingService()
        query_vector = embedder.embed(query_text)

        matches = await repo.most_similar_articles(query_vector, limit=top_k)

        if not matches:
            logger.info(
                "no_similar_articles_found",
                reason="no scored+embedded articles exist yet in the corpus",
            )
            return

        logger.info("query", text=query_text)
        for m in matches:
            logger.info(
                "similar_article",
                similarity=round(m.similarity, 4),
                sentiment=m.sentiment_label.value,
                confidence=round(m.sentiment_confidence, 3),
                title=m.title[:80],
            )

        label_counts = Counter(m.sentiment_label.value for m in matches)
        majority_label, majority_count = label_counts.most_common(1)[0]
        logger.info(
            "historical_precedent_summary",
            majority_sentiment=majority_label,
            agreement=f"{majority_count}/{len(matches)}",
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "Company announces record quarterly earnings"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(main(query, k))
