import pytest

from marketpulse.config.settings import Settings, get_settings


def test_settings_load_with_defaults_when_env_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.finnhub.api_key == ""
    assert settings.environment.value == "local"


def test_require_finnhub_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        settings.require_finnhub()


def test_require_finnhub_passes_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key-123")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
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
