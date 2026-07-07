"""Sanity checks for FinBertSentimentScorer against unambiguous inputs.

Not exhaustive accuracy testing (that's what a proper eval set is for) --
these exist to catch a genuinely broken model (wrong device, wrong label
mapping, garbage output) before it silently poisons the whole pipeline.
A model that can't distinguish "profits surge" from "shares collapse" from
neutral factual text is broken in a way worth failing loudly on.
"""

from __future__ import annotations

import pytest

from marketpulse.models.sentiment import FinBertSentimentScorer, SentimentLabel

pytestmark = pytest.mark.sanity


@pytest.fixture(scope="module")
def scorer() -> FinBertSentimentScorer:
    return FinBertSentimentScorer()


def test_clearly_positive_headline_scores_positive(scorer: FinBertSentimentScorer) -> None:
    result = scorer.score("Company profits surge 40% beating all analyst expectations")
    assert result.label == SentimentLabel.POSITIVE


def test_clearly_negative_headline_scores_negative(scorer: FinBertSentimentScorer) -> None:
    result = scorer.score("Company shares collapse amid fraud investigation and bankruptcy fears")
    assert result.label == SentimentLabel.NEGATIVE


def test_neutral_factual_headline_scores_neutral(scorer: FinBertSentimentScorer) -> None:
    result = scorer.score("Company will report quarterly earnings on Thursday")
    assert result.label == SentimentLabel.NEUTRAL


def test_probabilities_sum_to_approximately_one(scorer: FinBertSentimentScorer) -> None:
    result = scorer.score("Stock prices moved today")
    total = result.positive_prob + result.negative_prob + result.neutral_prob
    assert total == pytest.approx(1.0, abs=1e-4)


def test_rejects_empty_text(scorer: FinBertSentimentScorer) -> None:
    with pytest.raises(ValueError, match="empty text"):
        scorer.score("")


def test_score_many_matches_score_one_at_a_time(scorer: FinBertSentimentScorer) -> None:
    texts = ["Profits surge to record highs", "Shares collapse on fraud fears"]
    batch_results = scorer.score_many(texts)
    individual_results = [scorer.score(t) for t in texts]

    for batch_r, individual_r in zip(batch_results, individual_results, strict=True):
        assert batch_r.label == individual_r.label
