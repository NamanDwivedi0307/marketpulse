"""API response models.

Deliberately separate from the internal dataclasses/pydantic models used in
storage and ingestion (Quote, NewsArticle, SimilarArticle, etc.) -- an API
response shape is a public contract that should change deliberately, not
whenever an internal storage detail changes. Coupling them means a refactor
of internal storage silently breaks API consumers.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class QuoteResponse(BaseModel):
    symbol: str
    current_price: float
    change: float
    percent_change: float
    quoted_at: datetime


class NewsArticleResponse(BaseModel):
    uuid: str
    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    sentiment_label: str | None


class SimilarEventResponse(BaseModel):
    uuid: str
    title: str
    sentiment_label: str
    sentiment_confidence: float
    similarity: float


class HistoricalPrecedentResponse(BaseModel):
    query_text: str
    matches: list[SimilarEventResponse]
    majority_sentiment: str | None
    agreement_ratio: str | None


class ErrorResponse(BaseModel):
    detail: str

