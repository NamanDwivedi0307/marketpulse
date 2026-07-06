"""Typed models for data returned by the Marketaux news API.

Marketaux tags each article with zero or more "entities" (companies/tickers
it mentions), each carrying its own sentiment_score. A single article often
mentions multiple symbols with different relevance and sentiment per symbol
-- that per-entity structure is preserved here rather than flattened into
one sentiment-per-article, because the historical-event-matching engine
needs to know "how did this specific symbol's mention read," not just
"was this article generally positive."
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class NewsEntity(BaseModel):
    """One symbol/company mentioned within a news article."""

    symbol: str
    name: str
    sentiment_score: float | None = None
    # Marketaux's 0-1 relevance of this entity to the article as a whole --
    # a symbol mentioned in passing scores lower than the article's subject.
    match_score: float | None = Field(default=None, alias="match_score")

    model_config = {"populate_by_name": True}


class NewsArticle(BaseModel):
    """A single news article as returned by Marketaux's /news/all endpoint."""

    uuid: str
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    entities: list[NewsEntity] = Field(default_factory=list)

    def entity_for_symbol(self, symbol: str) -> NewsEntity | None:
        """Find this article's entity data for a specific symbol, if present.

        Returns None if the article doesn't actually mention that symbol --
        callers must handle that case explicitly rather than assuming every
        article returned by a symbol-filtered query definitely tags it
        (Marketaux's symbol filter is occasionally looser than expected).
        """
        for entity in self.entities:
            if entity.symbol.upper() == symbol.upper():
                return entity
        return None


class NewsResponse(BaseModel):
    """Parsed and validated response from a Marketaux news query."""

    articles: list[NewsArticle]

    @classmethod
    def from_marketaux_response(cls, payload: dict[str, object]) -> NewsResponse:
        raw_articles = payload.get("data")
        if raw_articles is None:
            raise ValueError(
                "Malformed Marketaux response: missing 'data' field. "
                f"Got keys: {list(payload.keys())}"
            )
        if not isinstance(raw_articles, list):
            raise ValueError(
                f"Malformed Marketaux response: 'data' should be a list, got {type(raw_articles)}"
            )

        articles = [NewsArticle.model_validate(item) for item in raw_articles]
        return cls(articles=articles)

