"""Loads the trained XGBoost model and produces live next-day return
predictions from recent quote history.

Deliberately mirrors the exact feature engineering in
scripts/train_forecast_model.py -- any drift between training-time and
inference-time feature computation silently produces wrong predictions
(the classic train/serve skew bug), so the feature list and formulas here
must stay identical to what the model was trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb

from marketpulse.ingestion.finnhub_models import Quote

logger = structlog.get_logger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "forecast_xgb.json"

FEATURE_COLUMNS = [
    "return_1d",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "volatility_10d",
    "high_low_range",
    "day_of_week",
    "rsi_14",
]

# Needs at least 20 prior days for the longest rolling window (momentum_20d)
# plus one more for the return itself to be defined.
MIN_QUOTES_REQUIRED = 21


class ForecastService:
    def __init__(self) -> None:
        self._model = xgb.XGBRegressor()
        self._model.load_model(str(MODEL_PATH))
        logger.info("forecast_model_loaded", path=str(MODEL_PATH))

    def predict_next_day_return(self, quotes: list[Quote]) -> float | None:
        """Predicted next-day percent return, or None if there isn't enough
        recent history to compute the required rolling features."""
        if len(quotes) < MIN_QUOTES_REQUIRED:
            return None

        df = pd.DataFrame(
            [
                {
                    "close": q.current_price,
                    "high": q.high_of_day,
                    "low": q.low_of_day,
                    "date": q.quoted_at,
                }
                for q in quotes
            ]
        )

        df["return_1d"] = df["close"].pct_change(1) * 100
        df["momentum_5d"] = df["close"].pct_change(5) * 100
        df["momentum_10d"] = df["close"].pct_change(10) * 100
        df["momentum_20d"] = df["close"].pct_change(20) * 100
        df["volatility_10d"] = df["return_1d"].rolling(10).std()
        df["high_low_range"] = (df["high"] - df["low"]) / df["close"] * 100
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        latest = df.iloc[[-1]][FEATURE_COLUMNS]
        if latest.isna().any(axis=None):
            return None

        prediction = self._model.predict(latest)[0]
        return float(prediction)
