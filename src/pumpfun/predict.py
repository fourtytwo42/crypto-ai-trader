"""Pump.fun model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.predict import forecast_next_horizon
from src.pumpfun.data import PUMPFUN_FEATURE_COLUMNS, PUMPFUN_NORMALIZED_COLUMNS
from src.pumpfun.db import get_pumpfun_db_manager
from src.pumpfun.models import PumpCandle1m, PumpToken
from src.pumpfun.recent_features import build_recent_feature_df
from src.prediction.model_loader import load_model_artifacts


@dataclass
class PumpfunPrediction:
    token_id: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_pct: float
    direction: str
    minutes: int
    confidence: float


def _load_recent_candles(token_id: str, limit: int) -> pd.DataFrame:
    db = get_pumpfun_db_manager()
    with db.session() as session:
        rows = (
            session.query(PumpCandle1m)
            .filter(PumpCandle1m.token_id == token_id)
            .order_by(PumpCandle1m.timestamp.desc())
            .limit(limit)
            .all()
        )
    if not rows:
        return pd.DataFrame()
    rows = list(reversed(rows))
    return pd.DataFrame(
        {
            "timestamp": [row.timestamp for row in rows],
            "open": [float(row.open) for row in rows],
            "high": [float(row.high) for row in rows],
            "low": [float(row.low) for row in rows],
            "close": [float(row.close) for row in rows],
            "volume_usd": [float(row.volume_usd) for row in rows],
            "trades": [int(row.trades) for row in rows],
        }
    )


def _load_token_meta(token_id: str) -> dict[str, object]:
    db = get_pumpfun_db_manager()
    with db.session() as session:
        token = session.query(PumpToken).filter(PumpToken.id == token_id).one_or_none()
    if token is None:
        return {}
    return {
        "created_timestamp": token.created_timestamp,
        "king_of_the_hill_timestamp": token.king_of_the_hill_timestamp,
        "completed": token.completed,
    }


def predict_pumpfun(
    token_id: str,
    model_dir: str | Path,
    minutes: int,
    context_length: int | None = None,
    target_mode: str = "sum",
) -> PumpfunPrediction:
    bundle = load_model_artifacts(model_dir)
    config_meta = bundle.metadata.get("config", {})
    model_horizon = int(config_meta.get("horizon", 10))
    context_length = context_length or int(config_meta.get("context_length", 336))

    if minutes > model_horizon and target_mode == "sum":
        raise ValueError(
            f"Requested {minutes} minutes, but model horizon is {model_horizon}. Retrain with a larger horizon."
        )
    if target_mode == "direct" and minutes != model_horizon:
        raise ValueError(
            f"Requested {minutes} minutes, but model predicts {model_horizon}-minute horizon."
        )

    candles_df = _load_recent_candles(token_id, context_length + model_horizon + 60)
    if candles_df.empty:
        raise ValueError("No candles available for token")

    token_meta = _load_token_meta(token_id)
    features_df, normalizer = build_recent_feature_df(candles_df, token_meta=token_meta)
    if features_df.empty:
        raise ValueError("No features available for token")

    history_df = features_df.tail(context_length).reset_index(drop=True)
    current_price = float(candles_df["close"].iloc[-1])

    raw_cols = [*PUMPFUN_FEATURE_COLUMNS, *PUMPFUN_NORMALIZED_COLUMNS, "log_close", "log_volume"]
    feature_cols = [
        col for col in dict.fromkeys(raw_cols) if col in history_df.columns and col != "return"
    ]
    preds = forecast_next_horizon(
        bundle.model,
        history_df,
        target_col="return_horizon" if target_mode == "direct" else "return",
        feature_cols=feature_cols,
        horizon=model_horizon,
    )
    preds = np.asarray(preds, dtype=float)
    if target_mode == "direct":
        pred_return = float(preds[-1])
    else:
        pred_return = float(np.sum(preds[:minutes]))

    predicted_price = float(current_price * np.exp(pred_return))
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100 if current_price else 0.0
    direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"
    confidence = float(1.0 / (1.0 + np.exp(-abs(actual_log_return) * 8)))

    return PumpfunPrediction(
        token_id=token_id,
        current_price=current_price,
        predicted_price=predicted_price,
        price_change=price_change,
        price_change_pct=price_change_pct,
        direction=direction,
        minutes=minutes,
        confidence=confidence,
    )
