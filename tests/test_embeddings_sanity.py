"""Sanity checks for EmbeddingService against unambiguous similarity cases.

Same rationale as test_sentiment_sanity.py -- catches a broken embedding
model (wrong device, garbage vectors, a version bump that silently changes
output dimension) rather than exhaustively validating embedding quality.
"""

from __future__ import annotations

import pytest

from marketpulse.models.embeddings import EMBEDDING_DIMENSION, EmbeddingService


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def embedder() -> EmbeddingService:
    return EmbeddingService()


def test_embedding_has_expected_dimension(embedder: EmbeddingService) -> None:
    vector = embedder.embed("Company reports strong quarterly earnings")
    assert len(vector) == EMBEDDING_DIMENSION


def test_similar_sentences_score_higher_than_unrelated_ones(embedder: EmbeddingService) -> None:
    base = embedder.embed("Semiconductor companies rally as chip demand surges")
    similar = embedder.embed("Chip stocks jump on strong AI-driven demand")
    unrelated = embedder.embed("Local bakery wins award for best sourdough bread")

    sim_to_similar = _cosine_similarity(base, similar)
    sim_to_unrelated = _cosine_similarity(base, unrelated)

    assert sim_to_similar > sim_to_unrelated


def test_identical_text_has_similarity_of_one(embedder: EmbeddingService) -> None:
    text = "Nvidia announces new AI chip architecture"
    a = embedder.embed(text)
    b = embedder.embed(text)
    assert _cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-5)


def test_rejects_empty_text(embedder: EmbeddingService) -> None:
    with pytest.raises(ValueError, match="empty text"):
        embedder.embed("")


def test_embed_many_matches_embed_one_at_a_time(embedder: EmbeddingService) -> None:
    texts = ["Company reports record profits", "Local weather turns cold this weekend"]
    batch_vectors = embedder.embed_many(texts)
    individual_vectors = [embedder.embed(t) for t in texts]

    for batch_v, individual_v in zip(batch_vectors, individual_vectors, strict=True):
        # Batched vs individual encoding can differ by tiny floating-point
        # noise depending on padding within the batch -- near-identical,
        # not bit-for-bit identical, is the correct bar here.
        assert _cosine_similarity(batch_v, individual_v) > 0.999
