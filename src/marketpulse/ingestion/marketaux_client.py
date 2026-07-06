"""Async client for the Marketaux news API.

Same design pattern as FinnhubClient: rate-limited, retries only on
transient failures, typed responses. The one real difference is budget --
Marketaux's free tier is 100 requests/DAY, not per-minute, so the rate
limiter here is configured with a 24-hour refill window. That budget is
scarce enough that every call should be deliberate; this client does not
attempt to paginate through all results automatically, since that could
burn the entire daily budget on one call site's mistake.
"""

from __future__ import annotations

from types import TracebackType

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from marketpulse.config.settings import Settings, get_settings
from marketpulse.ingestion.marketaux_models import NewsResponse
from marketpulse.ingestion.rate_limiter import TokenBucketRateLimiter

logger = structlog.get_logger(__name__)

_SECONDS_PER_DAY = 86400.0


class MarketauxError(Exception):
    """Base exception for all Marketaux client failures."""


class MarketauxAuthError(MarketauxError):
    """Raised on 401/403 -- never retried, since retrying won't fix a bad key."""


class MarketauxQuotaExceededError(MarketauxError):
    """Raised on 429 -- the daily request budget is exhausted.

    Never retried automatically: retrying a quota error just wastes more of
    tomorrow's budget checking today's exhaustion.
    """


class MarketauxTransientError(MarketauxError):
    """Raised on timeouts and 5xx responses -- safe to retry."""


class MarketauxClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._settings.require_marketaux()

        self._rate_limiter = TokenBucketRateLimiter(
            max_tokens=self._settings.marketaux.max_requests_per_day,
            refill_period_seconds=_SECONDS_PER_DAY,
        )
        self._http = httpx.AsyncClient(
            base_url=str(self._settings.marketaux.base_url),
            timeout=httpx.Timeout(10.0, connect=5.0),
        )

    async def __aenter__(self) -> MarketauxClient:
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
        retry=retry_if_exception_type((MarketauxTransientError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get(self, path: str, params: dict[str, str]) -> dict[str, object]:
        await self._rate_limiter.acquire()

        params = {**params, "api_token": self._settings.marketaux.api_key}
        log = logger.bind(path=path, params={k: v for k, v in params.items() if k != "api_token"})

        try:
            response = await self._http.get(path, params=params)
        except httpx.TimeoutException:
            log.warning("marketaux_request_timeout")
            raise

        if response.status_code in (401, 403):
            log.error("marketaux_auth_failed", status=response.status_code)
            raise MarketauxAuthError(
                f"Marketaux rejected the API key (status {response.status_code}). "
                "Check MARKETAUX_API_KEY in your .env file."
            )
        if response.status_code == 429:
            log.error("marketaux_quota_exceeded")
            raise MarketauxQuotaExceededError(
                "Marketaux daily request quota exhausted (free tier: 100/day). "
                "Wait until the quota resets or reduce polling frequency."
            )
        if response.status_code >= 500:
            log.warning("marketaux_server_error", status=response.status_code)
            raise MarketauxTransientError(
                f"Marketaux returned {response.status_code}, treating as transient"
            )
        if response.status_code != 200:
            raise MarketauxError(
                f"Unexpected Marketaux status {response.status_code}: {response.text[:200]}"
            )

        result: dict[str, object] = response.json()
        return result

    async def get_news_for_symbols(
        self,
        symbols: list[str],
        limit: int = 10,
    ) -> NewsResponse:
        """Fetch recent news mentioning any of the given symbols.

        symbols: e.g. ["AAPL", "MSFT"] -- Marketaux ORs these together.
        limit: max articles to return (free tier caps this fairly low).
        """
        payload = await self._get(
            "/news/all",
            params={
                "symbols": ",".join(symbols),
                "limit": str(limit),
                "language": "en",
            },
        )
        return NewsResponse.from_marketaux_response(payload)
