import os

import pytest
from pytest_httpx import HTTPXMock

from marketpulse.config.settings import FinnhubSettings, Settings
from marketpulse.ingestion.finnhub_client import (
    FinnhubAuthError,
    FinnhubClient,
    FinnhubNotFoundError,
    FinnhubTransientError,
)


def _settings_with_key() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        finnhub=FinnhubSettings(api_key="test-key", _env_file=None),  # type: ignore[call-arg]
    )


async def test_get_quote_parses_successful_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://finnhub.io/api/v1/quote?symbol=AAPL",
        json={
            "c": 195.89,
            "d": 1.23,
            "dp": 0.63,
            "h": 196.50,
            "l": 194.20,
            "o": 195.00,
            "pc": 194.66,
            "t": 1719648000,
        },
    )
    async with FinnhubClient(settings=_settings_with_key()) as client:
        quote = await client.get_quote("AAPL")
    assert quote.symbol == "AAPL"
    assert quote.current_price == 195.89


async def test_get_quote_raises_auth_error_on_401(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=401, json={"error": "invalid api key"})
    async with FinnhubClient(settings=_settings_with_key()) as client:
        with pytest.raises(FinnhubAuthError, match="rejected the API key"):
            await client.get_quote("AAPL")


async def test_get_quote_raises_not_found_on_404(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=404)
    async with FinnhubClient(settings=_settings_with_key()) as client:
        with pytest.raises(FinnhubNotFoundError):
            await client.get_quote("NOT_A_REAL_SYMBOL")


async def test_get_quote_retries_on_500_then_succeeds(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(status_code=500)
    httpx_mock.add_response(
        json={
            "c": 100.0,
            "d": 0.5,
            "dp": 0.5,
            "h": 101.0,
            "l": 99.0,
            "o": 100.0,
            "pc": 99.5,
            "t": 1719648000,
        }
    )
    async with FinnhubClient(settings=_settings_with_key()) as client:
        quote = await client.get_quote("AAPL")
    assert quote.current_price == 100.0


async def test_get_quote_exhausts_retries_and_raises(httpx_mock: HTTPXMock) -> None:
    for _ in range(3):
        httpx_mock.add_response(status_code=503)
    async with FinnhubClient(settings=_settings_with_key()) as client:
        with pytest.raises(FinnhubTransientError):
            await client.get_quote("AAPL")


async def test_get_candles_handles_no_data(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"s": "no_data"})
    async with FinnhubClient(settings=_settings_with_key()) as client:
        series = await client.get_candles("OBSCURE", "1", 0, 100)
    assert series.candles == []


@pytest.mark.skipif(
    not os.getenv("FINNHUB_API_KEY"),
    reason="Live Finnhub integration test requires a real FINNHUB_API_KEY in the environment",
)
async def test_live_get_quote_against_real_finnhub_api() -> None:
    """Actually hits the live Finnhub API. Requires a real free-tier key.

    Run with: FINNHUB_API_KEY=your_key uv run pytest -m "" tests/test_finnhub_client.py -k live
    """
    async with FinnhubClient() as client:
        quote = await client.get_quote("AAPL")
    assert quote.symbol == "AAPL"
    assert quote.current_price > 0


async def test_client_raises_clear_error_with_no_key() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        finnhub=FinnhubSettings(api_key="", _env_file=None),  # type: ignore[call-arg]
    )
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY is not set"):
        FinnhubClient(settings=settings)
