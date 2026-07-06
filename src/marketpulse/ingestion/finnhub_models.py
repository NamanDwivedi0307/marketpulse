"""Typed models for data returned by the Finnhub API.

Finnhub returns bare JSON with terse, undocumented-feeling field names
(c, h, l, o, pc, t). Wrapping that in named, validated models means every
other part of the codebase works with `quote.current_price`, not `data["c"]`,
and a shape change in the API surfaces as a clear validation error at the
ingestion boundary instead of a wrong number silently flowing into a model.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


class Quote(BaseModel):
    """A single real-time quote snapshot for one symbol."""

    symbol: str
    current_price: float = Field(alias="c")
    change: float = Field(alias="d")
    percent_change: float = Field(alias="dp")
    high_of_day: float = Field(alias="h")
    low_of_day: float = Field(alias="l")
    open_price: float = Field(alias="o")
    previous_close: float = Field(alias="pc")
    quoted_at: datetime = Field(alias="t")

    model_config = {"populate_by_name": True}

    @field_validator("quoted_at", mode="before")
    @classmethod
    def parse_unix_timestamp(cls, value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, int | float):
            return datetime.fromtimestamp(value, tz=UTC)
        raise ValueError(f"Expected unix timestamp, got {value!r}")

    @field_validator("current_price", "high_of_day", "low_of_day", "open_price")
    @classmethod
    def reject_zero_price(cls, value: float) -> float:
        # Finnhub returns all-zero payloads for symbols with no current
        # trading activity (e.g. delisted, or a market that's fully closed
        # with no cached quote). A zero price is not a real quote and should
        # never silently flow into a model as if it were one.
        if value == 0:
            raise ValueError(
                "Received zero price -- symbol likely has no active quote "
                "data (delisted, or market closed with no cache)"
            )
        return value


class Candle(BaseModel):
    """A single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleSeries(BaseModel):
    """A parsed and validated series of OHLCV bars for one symbol."""

    symbol: str
    resolution: str
    candles: list[Candle]

    @classmethod
    def from_finnhub_response(
        cls, symbol: str, resolution: str, payload: dict[str, object]
    ) -> CandleSeries:
        status = payload.get("s")
        if status == "no_data":
            return cls(symbol=symbol, resolution=resolution, candles=[])
        if status != "ok":
            raise ValueError(f"Unexpected Finnhub candle status: {status!r}")

        raw_fields = {
            "t": payload.get("t"),
            "o": payload.get("o"),
            "h": payload.get("h"),
            "l": payload.get("l"),
            "c": payload.get("c"),
            "v": payload.get("v"),
        }
        if not all(isinstance(field, list) for field in raw_fields.values()):
            raise ValueError(
                "Malformed Finnhub candle response: expected parallel arrays "
                "for t/o/h/l/c/v"
            )

        # Safe: every value was just confirmed to be a list above.
        timestamps: list[float] = raw_fields["t"]  # type: ignore[assignment]
        opens: list[float] = raw_fields["o"]  # type: ignore[assignment]
        highs: list[float] = raw_fields["h"]  # type: ignore[assignment]
        lows: list[float] = raw_fields["l"]  # type: ignore[assignment]
        closes: list[float] = raw_fields["c"]  # type: ignore[assignment]
        volumes: list[float] = raw_fields["v"]  # type: ignore[assignment]

        candles = [
            Candle(
                timestamp=datetime.fromtimestamp(ts, tz=UTC),
                open=o,
                high=h,
                low=low_,
                close=c,
                volume=v,
            )
            for ts, o, h, low_, c, v in zip(
                timestamps, opens, highs, lows, closes, volumes, strict=True
            )
        ]
        return cls(symbol=symbol, resolution=resolution, candles=candles)
