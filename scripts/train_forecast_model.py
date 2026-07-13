"""Trains an XGBoost model to predict next-day percent return per symbol.

Price-only features for v1 -- sentiment coverage currently spans only the
last few days of live-polled news, versus ~2 years of backfilled price
history, so joining sentiment in now would mean nulls for the vast majority
of training rows. Revisit once news_articles has enough historical depth
to cover a meaningful fraction of the price history.

Uses a time-based train/test split (not random) -- shuffling daily rows
before splitting would let the model train on data from *after* some test
dates, which is lookahead leakage and would produce misleadingly good
metrics that don't hold up on genuinely unseen future data.

Expected result: ~50% directional accuracy (coin-flip) on held-out data.
This is not a bug -- daily equity returns are close to a random walk, and
price-only technical features carry essentially no consistent next-day
directional signal, which is well established in finance literature (weak-
form market efficiency). This model's honest purpose is as the price-only
baseline in the fusion layer (see fusion_score.py), combined with the real
edge sources this project already has: sentiment (FinBERT) and historical
precedent (pgvector similarity + realized forward returns).

Usage:
    uv run python scripts/train_forecast_model.py AAPL MSFT GOOGL
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from marketpulse.config.settings import get_settings
from marketpulse.storage.pool import create_pool
from marketpulse.utils.logging import configure_logging

logger = structlog.get_logger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-day return forecast model")
    parser.add_argument("symbols", nargs="+", help="Symbols to train on (pooled into one model)")
    parser.add_argument(
        "--test-fraction", type=float, default=0.15, help="Fraction of most recent rows held out"
    )
    return parser.parse_args()


async def load_quotes(symbols: list[str]) -> pd.DataFrame:
    settings = get_settings()
    pool = await create_pool(settings.database)
    try:
        rows = await pool.fetch(
            """
            SELECT DISTINCT ON (symbol, quoted_at::date)
                symbol, current_price, high_of_day, low_of_day, open_price, quoted_at
            FROM quotes
            WHERE symbol = ANY($1::text[])
            ORDER BY symbol, quoted_at::date, quoted_at DESC
            """,
            symbols,
        )
    finally:
        await pool.close()
    return pd.DataFrame(
        [
            {
                "symbol": r["symbol"],
                "close": r["current_price"],
                "high": r["high_of_day"],
                "low": r["low_of_day"],
                "open": r["open_price"],
                "date": r["quoted_at"],
            }
            for r in rows
        ]
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds technical features per symbol, then concatenates back together.

    Grouping by symbol before computing rolling windows is required --
    otherwise a rolling mean would blend AAPL and MSFT prices at the
    boundary between symbols, producing nonsense features.
    """
    out_frames = []
    for _symbol, group in df.groupby("symbol"):
        g = group.sort_values("date").reset_index(drop=True).copy()

        g["return_1d"] = g["close"].pct_change(1) * 100
        g["momentum_5d"] = g["close"].pct_change(5) * 100
        g["momentum_10d"] = g["close"].pct_change(10) * 100
        g["momentum_20d"] = g["close"].pct_change(20) * 100
        g["volatility_10d"] = g["return_1d"].rolling(10).std()
        g["high_low_range"] = (g["high"] - g["low"]) / g["close"] * 100
        g["day_of_week"] = pd.to_datetime(g["date"]).dt.dayofweek

        delta = g["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        # Target: next day's return, computed now while still grouped by
        # symbol so it never leaks across the AAPL/MSFT boundary.
        g["target_next_return"] = g["close"].pct_change(1).shift(-1) * 100

        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    return result.dropna().reset_index(drop=True)


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


def time_based_split(
    df: pd.DataFrame, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_fraction))
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


async def main() -> None:
    args = parse_args()
    configure_logging()
    symbols = [s.upper() for s in args.symbols]

    raw = await load_quotes(symbols)
    if raw.empty:
        logger.error("no_quote_data", symbols=symbols)
        return

    featured = engineer_features(raw)
    logger.info("features_built", rows=len(featured))

    train_df, test_df = time_based_split(featured, args.test_fraction)
    logger.info("split_done", train_rows=len(train_df), test_rows=len(test_df))

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_next_return"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_next_return"]

    # Held-out validation slice (last 15% of *train*, still before test in
    # time) drives early stopping -- with ~1350 train rows and daily-return
    # targets that are close to pure noise, a deep/many-tree model memorizes
    # training data almost immediately. Shallow trees + strong regularization
    # + early stopping keeps the model from fitting noise it can't generalize.
    val_split = int(len(X_train) * 0.85)
    X_fit, X_val = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_fit, y_val = y_train.iloc[:val_split], y_train.iloc[val_split:]

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=10,
        random_state=42,
        early_stopping_rounds=20,
        eval_metric="rmse",
    )
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    logger.info("best_iteration", best_iteration=model.best_iteration)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    dir_acc = directional_accuracy(y_test.to_numpy(), preds)

    logger.info(
        "eval_complete",
        rmse=round(rmse, 4),
        directional_accuracy=round(dir_acc, 4),
        naive_baseline_note="0.5 = coin flip",
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "forecast_xgb.json"
    model.save_model(str(model_path))

    metadata = {
        "symbols": symbols,
        "feature_columns": FEATURE_COLUMNS,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "rmse": rmse,
        "directional_accuracy": dir_acc,
    }
    (MODEL_DIR / "forecast_xgb_metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("model_saved", path=str(model_path))


if __name__ == "__main__":
    asyncio.run(main())
