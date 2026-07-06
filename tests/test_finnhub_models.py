import pytest
from pydantic import ValidationError

from marketpulse.ingestion.finnhub_models import CandleSeries, Quote


def test_quote_parses_valid_finnhub_payload() -> None:
    payload = {
        "c": 195.89,
        "d": 1.23,
        "dp": 0.63,
        "h": 196.50,
        "l": 194.20,
        "o": 195.00,
        "pc": 194.66,
        "t": 1719648000,
        "symbol": "AAPL",
    }
    quote = Quote.model_validate(payload)
    assert quote.symbol == "AAPL"
    assert quote.current_price == 195.89
    assert quote.quoted_at.year == 2024


def test_quote_rejects_zero_price_as_invalid() -> None:
    payload = {
        "c": 0,
        "d": 0,
        "dp": 0,
        "h": 0,
        "l": 0,
        "o": 0,
        "pc": 0,
        "t": 1719648000,
        "symbol": "DELISTED",
    }
    with pytest.raises(ValidationError, match="zero price"):
        Quote.model_validate(payload)


def test_candle_series_parses_ok_response() -> None:
    payload = {
        "s": "ok",
        "t": [1719648000, 1719648060],
        "o": [195.0, 195.5],
        "h": [195.8, 195.9],
        "l": [194.9, 195.1],
        "c": [195.5, 195.7],
        "v": [1000, 1200],
    }
    series = CandleSeries.from_finnhub_response("AAPL", "1", payload)
    assert len(series.candles) == 2
    assert series.candles[0].close == 195.5


def test_candle_series_handles_no_data_status() -> None:
    payload = {"s": "no_data"}
    series = CandleSeries.from_finnhub_response("OBSCURE", "1", payload)
    assert series.candles == []


def test_candle_series_rejects_unexpected_status() -> None:
    payload = {"s": "error"}
    with pytest.raises(ValueError, match="Unexpected Finnhub candle status"):
        CandleSeries.from_finnhub_response("AAPL", "1", payload)


def test_candle_series_rejects_mismatched_array_lengths() -> None:
    payload = {
        "s": "ok",
        "t": [1719648000, 1719648060],
        "o": [195.0],  # length mismatch
        "h": [195.8, 195.9],
        "l": [194.9, 195.1],
        "c": [195.5, 195.7],
        "v": [1000, 1200],
    }
    with pytest.raises(ValueError):  # noqa: PT011 - zip(strict=True) raises plain ValueError
        CandleSeries.from_finnhub_response("AAPL", "1", payload)
