"""Centralized application configuration.

All environment-dependent values are read exactly once, here, and validated
at process startup. Nothing else in the codebase should call os.environ or
os.getenv directly -- if a module needs a setting, it imports get_settings().
This keeps configuration bugs (missing key, wrong type, typo'd env var name)
visible at boot time instead of surfacing as a cryptic failure three network
calls deep into a request.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class FinnhubSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FINNHUB_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", description="Finnhub API key, free tier")
    base_url: HttpUrl = Field(default=HttpUrl("https://finnhub.io/api/v1"))
    ws_url: str = Field(default="wss://ws.finnhub.io")
    # Free tier: 60 calls/minute. Kept conservative to leave headroom for
    # retries without tripping the provider's own rate limiter.
    max_requests_per_minute: int = Field(default=50, gt=0, le=60)


class AlphaVantageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPHA_VANTAGE_")

    api_key: str = Field(default="")
    base_url: HttpUrl = Field(default=HttpUrl("https://www.alphavantage.co/query"))
    # Free tier is 25 requests/day as of 2024 -- this is a fallback source,
    # not a primary feed. Treat it as scarce.
    max_requests_per_day: int = Field(default=25, gt=0)


class MarketauxSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MARKETAUX_")

    api_key: str = Field(default="")
    base_url: HttpUrl = Field(default=HttpUrl("https://api.marketaux.com/v1"))
    max_requests_per_day: int = Field(default=100, gt=0)


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATABASE_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    name: str = Field(default="marketpulse")
    user: str = Field(default="marketpulse")
    password: str = Field(default="")

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Field(default=Environment.LOCAL)
    log_level: str = Field(default="INFO")

    finnhub: FinnhubSettings = Field(default_factory=FinnhubSettings)
    alpha_vantage: AlphaVantageSettings = Field(default_factory=AlphaVantageSettings)
    marketaux: MarketauxSettings = Field(default_factory=MarketauxSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {value!r}")
        return normalized

    def require_finnhub(self) -> None:
        """Fail loudly and early if a code path needs Finnhub but no key is set.

        Called at the top of anything that actually hits the network, rather
        than letting an empty key turn into a 401 several layers down.
        """
        if not self.finnhub.api_key:
            raise RuntimeError(
                "FINNHUB_API_KEY is not set. Get a free key at "
                "https://finnhub.io/register and add it to your .env file."
            )

    def require_marketaux(self) -> None:
        if not self.marketaux.api_key:
            raise RuntimeError(
                "MARKETAUX_API_KEY is not set. Get a free key at "
                "https://www.marketaux.com/account/dashboard and add it to your .env file."
            )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    lru_cache means the .env file is parsed once per process, not once per
    call site. Tests that need different settings should call
    get_settings.cache_clear() after monkeypatching environment variables.
    """
    return Settings()
