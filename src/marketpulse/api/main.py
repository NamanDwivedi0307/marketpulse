"""MarketPulse read API.

Exposes what the ingestion/storage/ML pipeline has already computed --
latest quotes, recent news with sentiment, and historical event matching.
Deliberately read-only: ingestion stays script/poller-driven, so this API
has no write endpoints and no risk of a malformed request corrupting the
pipeline's data.

Run with:
    uv run uvicorn marketpulse.api.main:app --reload
"""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException

from marketpulse.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from marketpulse.api.schemas import (
    HistoricalPrecedentResponse,
    NewsArticleResponse,
    QuoteResponse,
    SimilarEventResponse,
)
from marketpulse.config.settings import get_settings
from marketpulse.models.embeddings import EmbeddingService
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.pool import create_pool
from marketpulse.storage.quote_repository import QuoteRepository

logger = structlog.get_logger(__name__)

# Populated at startup via the lifespan context below -- module-level state
# is the pragmatic choice here (FastAPI's own dependency-override pattern is
# more testable, but adds real complexity for a single-developer read API
# with no auth/multi-tenancy concerns yet).
_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pool = await create_pool(settings.database)
    await run_migrations(pool)

    _state["pool"] = pool
    _state["quote_repo"] = QuoteRepository(pool)
    _state["news_repo"] = NewsRepository(pool)
    # Embedding model loaded once at startup, not per-request -- loading it
    # per-request would add multi-second latency to every historical-match
    # query, defeating the purpose of an API.
    _state["embedder"] = EmbeddingService()

    logger.info("api_startup_complete")
    yield

    await pool.close()
    logger.info("api_shutdown_complete")


app = FastAPI(
    title="MarketPulse API",
    description="Real-time financial data, sentiment, and historical event matching.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(RateLimitMiddleware, max_requests=30, window_seconds=60.0)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/quotes/{symbol}", response_model=QuoteResponse)
async def get_latest_quote(symbol: str) -> QuoteResponse:
    quote_repo: QuoteRepository = _state["quote_repo"]  # type: ignore[assignment]
    quote = await quote_repo.latest_for_symbol(symbol.upper())
    if quote is None:
        raise HTTPException(
            status_code=404,
            detail=f"No quote data found for symbol '{symbol.upper()}'",
        )
    return QuoteResponse(
        symbol=quote.symbol,
        current_price=quote.current_price,
        change=quote.change,
        percent_change=quote.percent_change,
        quoted_at=quote.quoted_at,
    )


@app.get("/news/{symbol}", response_model=list[NewsArticleResponse])
async def get_recent_news(symbol: str, limit: int = 10) -> list[NewsArticleResponse]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")

    news_repo: NewsRepository = _state["news_repo"]  # type: ignore[assignment]
    results = await news_repo.sentiment_for_symbol(symbol.upper(), limit=limit)
    return [
        NewsArticleResponse(
            uuid=article.uuid,
            title=article.title,
            description=article.description,
            url=article.url,
            source=article.source,
            published_at=article.published_at,
            sentiment_label=label.value if label else None,
        )
        for article, label in results
    ]


@app.get("/historical-precedent", response_model=HistoricalPrecedentResponse)
async def get_historical_precedent(query: str, top_k: int = 5) -> HistoricalPrecedentResponse:
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20")

    embedder: EmbeddingService = _state["embedder"]  # type: ignore[assignment]
    news_repo: NewsRepository = _state["news_repo"]  # type: ignore[assignment]

    query_vector = embedder.embed(query)
    matches = await news_repo.most_similar_articles(query_vector, limit=top_k)

    if not matches:
        return HistoricalPrecedentResponse(
            query_text=query,
            matches=[],
            majority_sentiment=None,
            agreement_ratio=None,
        )

    label_counts = Counter(m.sentiment_label.value for m in matches)
    majority_label, majority_count = label_counts.most_common(1)[0]

    return HistoricalPrecedentResponse(
        query_text=query,
        matches=[
            SimilarEventResponse(
                uuid=m.uuid,
                title=m.title,
                sentiment_label=m.sentiment_label.value,
                sentiment_confidence=m.sentiment_confidence,
                similarity=m.similarity,
            )
            for m in matches
        ],
        majority_sentiment=majority_label,
        agreement_ratio=f"{majority_count}/{len(matches)}",
    )
