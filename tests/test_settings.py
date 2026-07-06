import pytest

from marketpulse.config.settings import FinnhubSettings, Settings, get_settings


def _settings_without_finnhub_key() -> Settings:
    """Settings with neither the top-level nor the nested Finnhub .env read.

    Needed because FinnhubSettings declares its own env_file=".env", so
    passing _env_file=None to the top-level Settings alone is not enough --
    the nested class would still pick up a real key from your local .env.
    """
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        finnhub=FinnhubSettings(api_key="", _env_file=None),  # type: ignore[call-arg]
    )


def test_settings_load_with_defaults_when_env_absent() -> None:
    settings = _settings_without_finnhub_key()
    assert settings.finnhub.api_key == ""
    assert settings.environment.value == "local"


def test_require_finnhub_raises_when_key_missing() -> None:
    settings = _settings_without_finnhub_key()
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        settings.require_finnhub()


def test_require_finnhub_passes_when_key_present() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        finnhub=FinnhubSettings(api_key="test-key-123", _env_file=None),  # type: ignore[call-arg]
    )
    settings.require_finnhub()  # should not raise


def test_invalid_log_level_rejected() -> None:
    with pytest.raises(ValueError, match="log_level"):
        Settings(_env_file=None, log_level="NOT_A_LEVEL")  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
