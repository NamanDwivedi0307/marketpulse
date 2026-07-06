import pytest

from marketpulse.ingestion.marketaux_models import NewsArticle, NewsResponse


def _sample_article(**overrides: object) -> dict[str, object]:
    base = {
        "uuid": "abc-123",
        "title": "Nvidia announces new chip architecture",
        "description": "Details of the announcement...",
        "url": "https://example.com/article",
        "source": "reuters.com",
        "published_at": "2024-06-10T14:30:00.000000Z",
        "entities": [
            {
                "symbol": "NVDA",
                "name": "NVIDIA Corporation",
                "sentiment_score": 0.82,
                "match_score": 0.95,
            }
        ],
    }
    base.update(overrides)
    return base


def test_article_parses_valid_payload() -> None:
    article = NewsArticle.model_validate(_sample_article())
    assert article.title == "Nvidia announces new chip architecture"
    assert len(article.entities) == 1
    assert article.entities[0].symbol == "NVDA"


def test_entity_for_symbol_finds_case_insensitive_match() -> None:
    article = NewsArticle.model_validate(_sample_article())
    entity = article.entity_for_symbol("nvda")
    assert entity is not None
    assert entity.sentiment_score == 0.82


def test_entity_for_symbol_returns_none_when_absent() -> None:
    article = NewsArticle.model_validate(_sample_article())
    assert article.entity_for_symbol("AAPL") is None


def test_article_handles_no_entities() -> None:
    article = NewsArticle.model_validate(_sample_article(entities=[]))
    assert article.entities == []
    assert article.entity_for_symbol("NVDA") is None


def test_news_response_parses_full_payload() -> None:
    payload = {"data": [_sample_article(), _sample_article(uuid="def-456")]}
    response = NewsResponse.from_marketaux_response(payload)
    assert len(response.articles) == 2


def test_news_response_rejects_missing_data_field() -> None:
    with pytest.raises(ValueError, match="missing 'data' field"):
        NewsResponse.from_marketaux_response({"meta": {}})


def test_news_response_rejects_non_list_data() -> None:
    with pytest.raises(ValueError, match="should be a list"):
        NewsResponse.from_marketaux_response({"data": "not a list"})
