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

from marketpulse.api.middleware import (
    ApiKeyAuthMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from marketpulse.api.schemas import (
    FusionScoreResponse,
    HistoricalPrecedentResponse,
    NewsArticleResponse,
    QuoteResponse,
    SimilarEventResponse,
)
from marketpulse.config.settings import get_settings
from marketpulse.models.embeddings import EmbeddingService
from marketpulse.models.forecast import ForecastService
from marketpulse.models.fusion_score import compute_fusion_score
from marketpulse.storage.migrator import run_migrations
from marketpulse.storage.news_repository import NewsRepository
from marketpulse.storage.outcome_repository import OutcomeRepository
from marketpulse.storage.pool import create_pool
from marketpulse.storage.quote_repository import QuoteRepository

logger = structlog.get_logger(__name__)

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pool = await create_pool(settings.database)
    await run_migrations(pool)

    _state["pool"] = pool
    _state["quote_repo"] = QuoteRepository(pool)
    _state["news_repo"] = NewsRepository(pool)
    _state["outcome_repo"] = OutcomeRepository(pool)
    _state["embedder"] = EmbeddingService()
    _state["forecast_service"] = ForecastService()

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
app.add_middleware(ApiKeyAuthMiddleware, expected_key=get_settings().api.api_key)
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
async def get_historical_precedent(
    query: str,
    top_k: int = 5,
    symbol: str | None = None,
    horizon_minutes: int = 1440,
) -> HistoricalPrecedentResponse:
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 20")

    embedder: EmbeddingService = _state["embedder"]  # type: ignore[assignment]
    news_repo: NewsRepository = _state["news_repo"]  # type: ignore[assignment]
    outcome_repo: OutcomeRepository = _state["outcome_repo"]  # type: ignore[assignment]

    query_vector = embedder.embed(query)
    matches = await news_repo.most_similar_articles(query_vector, limit=top_k)

    if not matches:
        return HistoricalPrecedentResponse(
            query_text=query,
            matches=[],
            majority_sentiment=None,
            agreement_ratio=None,
            average_return_pct=None,
            return_sample_size=None,
        )

    label_counts = Counter(m.sentiment_label.value for m in matches)
    majority_label, majority_count = label_counts.most_common(1)[0]

    average_return_pct: float | None = None
    return_sample_size: int | None = None
    if symbol:
        match_uuids = [m.uuid for m in matches]
        average_return_pct = await outcome_repo.average_return_for_similar(
            match_uuids, symbol.upper(), horizon_minutes
        )
        if average_return_pct is not None:
            return_sample_size = len(match_uuids)

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
        average_return_pct=average_return_pct,
        return_sample_size=return_sample_size,
    )



@app.get("/fusion-score/{symbol}", response_model=FusionScoreResponse)
async def get_fusion_score(symbol: str) -> FusionScoreResponse:
    """Combines the three independently-built signals -- FinBERT sentiment
    on recent news, realized historical-precedent returns, and the price-only
    forecast model -- into one directional score for the symbol.

    Any signal that isn't available (no recent news, no similar historical
    articles, insufficient quote history for the forecast) is simply
    excluded from the blend rather than treated as zero/neutral -- see
    fusion_score.py for why that distinction matters.
    """
    symbol = symbol.upper()
    news_repo: NewsRepository = _state["news_repo"]  # type: ignore[assignment]
    outcome_repo: OutcomeRepository = _state["outcome_repo"]  # type: ignore[assignment]
    quote_repo: QuoteRepository = _state["quote_repo"]  # type: ignore[assignment]
    forecast_service: ForecastService = _state["forecast_service"]  # type: ignore[assignment]

    recent_news = await news_repo.sentiment_for_symbol(symbol, limit=1)
    sentiment_label = recent_news[0][1] if recent_news else None
    # sentiment_for_symbol doesn't return confidence directly -- article's
    # own recorded confidence isn't exposed by that method, so we treat a
    # present label as reasonably confident. Revisit if sentiment_confidence
    # becomes available on this query path.
    sentiment_confidence = 0.75 if sentiment_label else None

    average_precedent_return_pct: float | None = None
    # Historical precedent needs an embedding to search by -- reuse the most
    # recent article's title as the query text via the embedder already in
    # _state, mirroring how /historical-precedent does it.
    embedder: EmbeddingService = _state["embedder"]  # type: ignore[assignment]
    if recent_news:
        recent_article, _ = recent_news[0]
        query_vector = embedder.embed(recent_article.title)
        similar_articles = await news_repo.most_similar_articles(query_vector, limit=5)
        if similar_articles:
            match_uuids = [m.uuid for m in similar_articles]
            average_precedent_return_pct = await outcome_repo.average_return_for_similar(
                match_uuids, symbol, horizon_minutes=1440
            )

    recent_quotes = await quote_repo.recent_for_symbol(symbol, limit=30)
    forecast_return_pct = forecast_service.predict_next_day_return(recent_quotes)

    result = compute_fusion_score(
        symbol=symbol,
        sentiment_label=sentiment_label,
        sentiment_confidence=sentiment_confidence,
        average_precedent_return_pct=average_precedent_return_pct,
        forecast_return_pct=forecast_return_pct,
    )

    return FusionScoreResponse(
        symbol=result.symbol,
        fusion_score=result.fusion_score,
        label=result.label,
        confidence=result.confidence,
        sentiment_component=result.sentiment_component,
        precedent_component=result.precedent_component,
        forecast_component=result.forecast_component,
    )
