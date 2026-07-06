import os

import pytest
from pytest_httpx import HTTPXMock

from marketpulse.config.settings import MarketauxSettings, Settings
from marketpulse.ingestion.marketaux_client import (
    MarketauxAuthError,
    MarketauxClient,
    MarketauxQuotaExceededError,
    MarketauxTransientError,
)


def _settings_with_key() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        marketaux=MarketauxSettings(api_key="test-key", _env_file=None),  # type: ignore[call-arg]
    )


def _sample_response() -> dict[str, object]:
    return {
        "data": [
            {
                "uuid": "abc-123",
                "title": "Nvidia announces new chip architecture",
                "description": "Details...",
                "url": "https://example.com/article",
                "source": "reuters.com",
                "published_at": "2024-06-10T14:30:00.000000Z",
                "entities": [
                    {"symbol": "NVDA", "name": "NVIDIA Corporation", "sentiment_score": 0.82}
                ],
            }
        ]
    }


async def test_get_news_parses_successful_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json=_sample_response())
    async with MarketauxClient(settings=_settings_with_key()) as client:
        response = await client.get_news_for_symbols(["NVDA"])
    assert len(response.articles) == 1
    assert response.articles[0].entity_for_symbol("NVDA") is not None


async def test_get_news_raises_auth_error_on_401(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=401)
    async with MarketauxClient(settings=_settings_with_key()) as client:
        with pytest.raises(MarketauxAuthError):
            await client.get_news_for_symbols(["NVDA"])


async def test_get_news_raises_quota_error_on_429_without_retry(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=429)
    async with MarketauxClient(settings=_settings_with_key()) as client:
        with pytest.raises(MarketauxQuotaExceededError):
            await client.get_news_for_symbols(["NVDA"])
    # Only one request should have been made -- quota errors are not retried.
    assert len(httpx_mock.get_requests()) == 1


async def test_get_news_retries_on_500_then_succeeds(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(json=_sample_response())
    async with MarketauxClient(settings=_settings_with_key()) as client:
        response = await client.get_news_for_symbols(["NVDA"])
    assert len(response.articles) == 1


async def test_get_news_exhausts_retries_and_raises(httpx_mock: HTTPXMock) -> None:
    for _ in range(3):
        httpx_mock.add_response(status_code=503)
    async with MarketauxClient(settings=_settings_with_key()) as client:
        with pytest.raises(MarketauxTransientError):
            await client.get_news_for_symbols(["NVDA"])


async def test_client_raises_clear_error_with_no_key() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        marketaux=MarketauxSettings(api_key="", _env_file=None),  # type: ignore[call-arg]
    )
    with pytest.raises(RuntimeError, match="MARKETAUX_API_KEY is not set"):
        MarketauxClient(settings=settings)


@pytest.mark.skipif(
    not os.getenv("MARKETAUX_API_KEY"),
    reason="Live Marketaux integration test requires a real MARKETAUX_API_KEY",
)
async def test_live_get_news_against_real_marketaux_api() -> None:
    """Actually hits the live Marketaux API. Requires a real free-tier key.

    Run with: MARKETAUX_API_KEY=your_key uv run pytest -m "" tests/test_marketaux_client.py -k live
    """
    async with MarketauxClient() as client:
        response = await client.get_news_for_symbols(["AAPL"], limit=3)
    assert isinstance(response.articles, list)
