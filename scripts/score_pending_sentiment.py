"""Scores sentiment for all news articles that haven't been scored yet.

Run this periodically (e.g. after the news poller adds new articles) to
keep the sentiment backlog from growing. Not folded into NewsPoller itself
because model loading is slow (several seconds) and heavy (holds GPU/CPU
memory for the process lifetime) -- keeping it a separate, on-demand step
means the news poller stays lightweight and this can run on its own
schedule, or on a machine with a GPU while the poller runs elsewhere.

Usage:
    uv run python scripts/score_pending_sentiment.py --batch-size 50
"""

from __future__ import annotations

import argparse
import asyncio

import structlog

from marketpulse.config.settings import get_settings
from marketpulse.models.sentiment import FinBertSentimentScorer
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.pool import create_pool
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score pending news sentiment with FinBERT")
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Max articles to score per run"
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
        pending = await repo.unscored_articles(limit=args.batch_size)
        if not pending:
            logger.info("no_pending_articles_to_score")
            return

        logger.info("scoring_starting", article_count=len(pending))

        scorer = FinBertSentimentScorer()

        # Score title + description together -- the headline alone often
        # lacks enough context (e.g. "Company X reports Q3 results" is
        # neutral on its own; the description usually carries the actual
        # sentiment-bearing detail like "beating estimates by 12%").
        texts = [f"{a.title}. {a.description}" for a in pending]
        scores = scorer.score_many(texts)

        for article, score in zip(pending, scores, strict=True):
            await repo.save_sentiment(article.uuid, score)
            logger.info(
                "article_scored",
                uuid=article.uuid,
                title=article.title[:60],
                label=score.label.value,
                confidence=round(score.confidence, 3),
            )

        logger.info("scoring_complete", article_count=len(pending))
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
