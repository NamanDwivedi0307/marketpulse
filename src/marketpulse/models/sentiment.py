"""FinBERT-based sentiment scoring for financial news text.

Uses ProsusAI/finbert -- a BERT variant fine-tuned specifically on financial
text, which matters here: general-purpose sentiment models routinely
misread financial language (e.g. "shares fell" reads as neutral/factual to
a generic model but is clearly negative in a financial context; "beat
expectations" is positive despite containing no obviously positive words).

The model is loaded once per process, not per call -- loading FinBERT takes
real time (weight download on first run, then disk read + GPU/CPU transfer
on every subsequent process start), so a module-level singleton avoids
paying that cost per article scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import structlog
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = structlog.get_logger(__name__)

_MODEL_NAME = "ProsusAI/finbert"

# FinBERT's label order as published by the model card -- this is fixed by
# the model's training, not something to reorder without checking the
# actual model config first if this is ever changed.
_LABELS = ["positive", "negative", "neutral"]


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SentimentScore:
    label: SentimentLabel
    confidence: float
    positive_prob: float
    negative_prob: float
    neutral_prob: float


class FinBertSentimentScorer:
    def __init__(self) -> None:
        logger.info("finbert_loading", model=_MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        logger.info("finbert_loaded", device=self._device)

    def score(self, text: str) -> SentimentScore:
        if not text or not text.strip():
            raise ValueError("Cannot score empty text")

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            # FinBERT's underlying BERT has a 512-token context window --
            # long articles get truncated, not rejected. Good enough for
            # headline + description; revisit if scoring full article bodies.
            max_length=512,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].tolist()

        prob_by_label = dict(zip(_LABELS, probs, strict=True))
        top_label = max(prob_by_label, key=lambda k: prob_by_label[k])

        return SentimentScore(
            label=SentimentLabel(top_label),
            confidence=prob_by_label[top_label],
            positive_prob=prob_by_label["positive"],
            negative_prob=prob_by_label["negative"],
            neutral_prob=prob_by_label["neutral"],
        )

    def score_many(self, texts: list[str]) -> list[SentimentScore]:
        # Batched inference is meaningfully faster than a Python-level loop
        # calling score() per text, especially on GPU where the overhead per
        # forward pass otherwise dominates for short financial headlines.
        non_empty = [t for t in texts if t and t.strip()]
        if len(non_empty) != len(texts):
            raise ValueError("Cannot score empty text in batch")
        if not texts:
            return []

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).tolist()

        results = []
        for prob_row in probs:
            prob_by_label = dict(zip(_LABELS, prob_row, strict=True))
            top_label = max(prob_by_label, key=lambda k: prob_by_label[k])
            results.append(
                SentimentScore(
                    label=SentimentLabel(top_label),
                    confidence=prob_by_label[top_label],
                    positive_prob=prob_by_label["positive"],
                    negative_prob=prob_by_label["negative"],
                    neutral_prob=prob_by_label["neutral"],
                )
            )
        return results

