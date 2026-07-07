"""Sentence embedding generation for semantic similarity search.

Uses all-MiniLM-L6-v2 -- small (80MB), fast, and a standard, well-validated
choice for semantic similarity tasks. A larger/fancier embedding model would
cost meaningfully more compute per article for a task (nearest-neighbor
lookup over financial headlines) where MiniLM's quality ceiling is not the
bottleneck; the actual bottleneck is having enough historical data to match
against in the first place.

Same singleton-per-process pattern as FinBertSentimentScorer -- model
loading has real cost, so this should be constructed once and reused.
"""

from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # fixed by the model; must match the DB column


class EmbeddingService:
    def __init__(self) -> None:
        logger.info("embedding_model_loading", model=_MODEL_NAME)
        self._model = SentenceTransformer(_MODEL_NAME)
        logger.info("embedding_model_loaded")

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()  # type: ignore[no-any-return]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not t or not t.strip() for t in texts):
            raise ValueError("Cannot embed empty text in batch")
        vectors = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]
