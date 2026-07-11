"""Tests for the API route handlers.

Calls the route functions directly with a mocked _state rather than
spinning up the full ASGI app + lifespan (which would require a real DB and
a real embedding model load) -- this tests the actual business logic
(status codes, response shaping, validation) that's specific to this layer,
while the underlying repository/embedding logic is already covered
elsewhere (test_storage_integration.py, test_news_repository_integration.py).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from marketpulse.api import main as api_main
from marketpulse.ingestion.finnhub_models import Quote
from marketpulse.ingestion.marketaux_models import NewsArticle
from marketpulse.models.sentiment import SentimentLabel
from marketpulse.storage.news_repository import SimilarArticle


def _quote(symbol: str = "AAPL", price: float = 150.0) -> Quote:
    return Quote(
        symbol=symbol, c=price, d=1.0, dp=0.5,
        h=price + 1, l=price - 1, o=price, pc=price - 0.5,
        t=datetime.now(UTC),
    )


def _article(uuid: str = "a1") -> NewsArticle:
    return NewsArticle(
        uuid=uuid, title="Test title", description="Test desc",
        url="https://example.com", source="test", published_at=datetime.now(UTC),
        entities=[],
    )


async def test_health_returns_ok() -> None:
    result = await api_main.health()
    assert result == {"status": "ok"}


async def test_get_latest_quote_returns_data_when_found() -> None:
    api_main._state["quote_repo"] = AsyncMock(latest_for_symbol=AsyncMock(return_value=_quote()))
    result = await api_main.get_latest_quote("aapl")
    assert result.symbol == "AAPL"
    assert result.current_price == 150.0


async def test_get_latest_quote_raises_404_when_not_found() -> None:
    api_main._state["quote_repo"] = AsyncMock(latest_for_symbol=AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_latest_quote("NOSUCHSYMBOL")
    assert exc_info.value.status_code == 404


async def test_get_recent_news_rejects_invalid_limit() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_recent_news("AAPL", limit=0)
    assert exc_info.value.status_code == 422

    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_recent_news("AAPL", limit=101)
    assert exc_info.value.status_code == 422


async def test_get_recent_news_maps_sentiment_label_correctly() -> None:
    api_main._state["news_repo"] = AsyncMock(
        sentiment_for_symbol=AsyncMock(
            return_value=[(_article(), SentimentLabel.POSITIVE), (_article("a2"), None)]
        )
    )
    result = await api_main.get_recent_news("AAPL")
    assert result[0].sentiment_label == "positive"
    assert result[1].sentiment_label is None


async def test_historical_precedent_rejects_empty_query() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_historical_precedent(query="   ")
    assert exc_info.value.status_code == 422


async def test_historical_precedent_rejects_invalid_top_k() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await api_main.get_historical_precedent(query="test", top_k=0)
    assert exc_info.value.status_code == 422


async def test_historical_precedent_returns_empty_result_when_no_matches() -> None:
    api_main._state["embedder"] = AsyncMock(embed=lambda text: [0.1] * 384)
    api_main._state["news_repo"] = AsyncMock(most_similar_articles=AsyncMock(return_value=[]))
    api_main._state["outcome_repo"] = AsyncMock()

    result = await api_main.get_historical_precedent(query="something novel")
    assert result.matches == []
    assert result.majority_sentiment is None
    assert result.agreement_ratio is None
    assert result.average_return_pct is None
    assert result.return_sample_size is None


async def test_historical_precedent_computes_majority_and_agreement() -> None:
    api_main._state["embedder"] = AsyncMock(embed=lambda text: [0.1] * 384)
    api_main._state["news_repo"] = AsyncMock(
        most_similar_articles=AsyncMock(
            return_value=[
                SimilarArticle("u1", "t1", SentimentLabel.NEGATIVE, 0.9, 0.8),
                SimilarArticle("u2", "t2", SentimentLabel.NEGATIVE, 0.8, 0.6),
                SimilarArticle("u3", "t3", SentimentLabel.POSITIVE, 0.7, 0.3),
            ]
        )
    )
    api_main._state["outcome_repo"] = AsyncMock()

    result = await api_main.get_historical_precedent(query="chip stocks rally")
    assert result.majority_sentiment == "negative"
    assert result.agreement_ratio == "2/3"
    assert len(result.matches) == 3
    # no symbol passed -> outcome lookup skipped entirely
    assert result.average_return_pct is None
    assert result.return_sample_size is None


async def test_historical_precedent_includes_average_return_when_symbol_given() -> None:
    api_main._state["embedder"] = AsyncMock(embed=lambda text: [0.1] * 384)
    api_main._state["news_repo"] = AsyncMock(
        most_similar_articles=AsyncMock(
            return_value=[
                SimilarArticle("u1", "t1", SentimentLabel.NEGATIVE, 0.9, 0.8),
                SimilarArticle("u2", "t2", SentimentLabel.POSITIVE, 0.7, 0.3),
            ]
        )
    )
    api_main._state["outcome_repo"] = AsyncMock(
        average_return_for_similar=AsyncMock(return_value=-1.21)
    )

    result = await api_main.get_historical_precedent(
        query="chip stocks rally", symbol="AAPL", horizon_minutes=1440
    )
    assert result.average_return_pct == -1.21
    assert result.return_sample_size == 2
    api_main._state["outcome_repo"].average_return_for_similar.assert_awaited_once_with(
        ["u1", "u2"], "AAPL", 1440
    )
