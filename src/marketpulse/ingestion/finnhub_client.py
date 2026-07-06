"""Async client for the Finnhub REST API.

Design choices worth knowing before extending this:

- Retries only happen on transient failures (timeouts, connection errors,
  5xx). A 401 or 429 is never retried automatically -- a bad key retried
  five times just burns budget for the same failure, and a 429 needs the
  rate limiter fixed, not a retry loop papering over it.
- The rate limiter is applied before every request, not just when a 429 is
  seen. Waiting for a 429 to happen means we already wasted a call.
- All responses are parsed into pydantic models before being returned. If
  Finnhub changes their response shape, this raises a clear ValidationError
  at the call site instead of a KeyError somewhere else in the codebase.
"""

from __future__ import annotations

from types import TracebackType

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from marketpulse.config.settings import Settings, get_settings
from marketpulse.ingestion.finnhub_models import CandleSeries, Quote
from marketpulse.ingestion.rate_limiter import TokenBucketRateLimiter

logger = structlog.get_logger(__name__)


class FinnhubError(Exception):
    """Base exception for all Finnhub client failures."""


class FinnhubAuthError(FinnhubError):
    """Raised on 401/403 -- never retried, since retrying won't fix a bad key."""


class FinnhubNotFoundError(FinnhubError):
    """Raised on 404 -- the symbol doesn't exist or isn't covered."""


class FinnhubTransientError(FinnhubError):
    """Raised on timeouts and 5xx responses -- safe to retry."""


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, FinnhubTransientError | httpx.TimeoutException)


class FinnhubClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._settings.require_finnhub()

        self._rate_limiter = TokenBucketRateLimiter(
            max_tokens=self._settings.finnhub.max_requests_per_minute,
            refill_period_seconds=60.0,
        )
        self._http = httpx.AsyncClient(
            base_url=str(self._settings.finnhub.base_url),
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={"X-Finnhub-Token": self._settings.finnhub.api_key},
        )

    async def __aenter__(self) -> FinnhubClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    @retry(
        retry=retry_if_exception_type((FinnhubTransientError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, str]) -> dict[str, object]:
        await self._rate_limiter.acquire()

        log = logger.bind(path=path, params=params)
        try:
            response = await self._http.get(path, params=params)
        except httpx.TimeoutException:
            log.warning("finnhub_request_timeout")
            raise

        if response.status_code in (401, 403):
            log.error("finnhub_auth_failed", status=response.status_code)
            raise FinnhubAuthError(
                f"Finnhub rejected the API key (status {response.status_code}). "
                "Check FINNHUB_API_KEY in your .env file."
            )
        if response.status_code == 404:
            raise FinnhubNotFoundError(f"Finnhub has no data at {path} for {params}")
        if response.status_code >= 500:
            log.warning("finnhub_server_error", status=response.status_code)
            raise FinnhubTransientError(
                f"Finnhub returned {response.status_code}, treating as transient"
            )
        if response.status_code != 200:
            raise FinnhubError(
                f"Unexpected Finnhub status {response.status_code}: {response.text[:200]}"
            )

        result: dict[str, object] = response.json()
        return result

    async def get_quote(self, symbol: str) -> Quote:
        """Fetch the current real-time quote for a symbol.

        Symbol must be a Finnhub-recognized ticker, e.g. "AAPL", "MSFT".
        """
        payload = await self._get("/quote", params={"symbol": symbol})
        payload["symbol"] = symbol
        return Quote.model_validate(payload)

    async def get_candles(
        self,
        symbol: str,
        resolution: str,
        from_unix: int,
        to_unix: int,
    ) -> CandleSeries:
        """Fetch historical OHLCV candles.

        resolution: one of "1", "5", "15", "30", "60", "D", "W", "M"
        from_unix / to_unix: unix timestamps in seconds (UTC)
        """
        payload = await self._get(
            "/stock/candle",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "from": str(from_unix),
                "to": str(to_unix),
            },
        )
        return CandleSeries.from_finnhub_response(symbol, resolution, payload)
