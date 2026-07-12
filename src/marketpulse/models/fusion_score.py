"""Combines sentiment, historical precedent, and price forecast into a
single directional score per symbol.

Design choice: this is a weighted blend, not a learned meta-model. With
three heterogeneous, individually-noisy signals and no labeled "correct
fusion weight" dataset to train on, a hand-set weighting is more honest
than dressing up an arbitrary choice as learned. Weights are documented
inline and easy to revisit once there's enough live data to actually
validate a learned combiner against.

Score range: -1.0 (strongly bearish) to +1.0 (strongly bullish). Each
component is normalized to that same range before blending, so no single
signal dominates just because its native units happen to be larger.
"""

from __future__ import annotations

from dataclasses import dataclass

from marketpulse.models.sentiment import SentimentLabel

# Weights sum to 1.0. Sentiment gets the largest weight since FinBERT is a
# well-validated, purpose-built signal; historical precedent next since it's
# grounded in realized outcomes (not just a proxy); price forecast last and
# lightest, since the price-only model was shown to have ~no standalone
# edge (see train_forecast_model.py) -- it's included for completeness and
# future improvement, not because it's currently a strong signal.
SENTIMENT_WEIGHT = 0.5
PRECEDENT_WEIGHT = 0.35
FORECAST_WEIGHT = 0.15

# Forecast returns are typically small in magnitude (well under 1%) relative
# to precedent returns (which can be several percent) -- this scales the
# forecast into a comparable range before blending. Not a statistically
# derived constant, just a reasonable order-of-magnitude normalizer.
FORECAST_NORMALIZATION_PCT = 2.0
PRECEDENT_NORMALIZATION_PCT = 5.0


@dataclass(frozen=True)
class FusionScore:
    symbol: str
    fusion_score: float  # -1.0 to +1.0
    sentiment_component: float | None
    precedent_component: float | None
    forecast_component: float | None
    label: str  # "bullish" | "bearish" | "neutral"
    confidence: float  # 0.0 to 1.0, based on how many signals were available


def _sentiment_to_score(label: SentimentLabel | None, confidence: float | None) -> float | None:
    if label is None or confidence is None:
        return None
    if label == SentimentLabel.NEUTRAL:
        return 0.0
    signed = 1.0 if label == SentimentLabel.POSITIVE else -1.0
    return signed * confidence


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute_fusion_score(
    symbol: str,
    sentiment_label: SentimentLabel | None,
    sentiment_confidence: float | None,
    average_precedent_return_pct: float | None,
    forecast_return_pct: float | None,
) -> FusionScore:
    sentiment_component = _sentiment_to_score(sentiment_label, sentiment_confidence)

    precedent_component: float | None = None
    if average_precedent_return_pct is not None:
        precedent_component = _clip(average_precedent_return_pct / PRECEDENT_NORMALIZATION_PCT)

    forecast_component: float | None = None
    if forecast_return_pct is not None:
        forecast_component = _clip(forecast_return_pct / FORECAST_NORMALIZATION_PCT)

    components = [
        (sentiment_component, SENTIMENT_WEIGHT),
        (precedent_component, PRECEDENT_WEIGHT),
        (forecast_component, FORECAST_WEIGHT),
    ]
    available = [(val, weight) for val, weight in components if val is not None]

    if not available:
        return FusionScore(
            symbol=symbol,
            fusion_score=0.0,
            sentiment_component=None,
            precedent_component=None,
            forecast_component=None,
            label="neutral",
            confidence=0.0,
        )

    # Renormalize weights over only the signals actually available, so a
    # missing signal doesn't silently drag the score toward zero just
    # because its weight went unused.
    weight_sum = sum(w for _, w in available)
    fusion_score = sum(val * w for val, w in available) / weight_sum
    fusion_score = _clip(fusion_score)

    if fusion_score > 0.15:
        label = "bullish"
    elif fusion_score < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    # Confidence reflects signal coverage, not statistical certainty --
    # a score built from all 3 signals is more trustworthy than one built
    # from a single available signal, independent of what that score is.
    confidence = weight_sum

    return FusionScore(
        symbol=symbol,
        fusion_score=fusion_score,
        sentiment_component=sentiment_component,
        precedent_component=precedent_component,
        forecast_component=forecast_component,
        label=label,
        confidence=confidence,
    )
