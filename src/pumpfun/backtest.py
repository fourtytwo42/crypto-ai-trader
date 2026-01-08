"""Pump.fun backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.forecasting.predict import forecast_next_horizon
from src.pumpfun.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from src.pumpfun.db import get_pumpfun_db_manager
from src.prediction.model_loader import load_model_artifacts


@dataclass
class PumpfunBacktestResult:
    mae: float
    rmse: float
    smape: float
    direction_accuracy: float
    price_accuracy_pct: float
    samples: int


def _load_holdout_tokens(model_dir: str | Path) -> list[str]:
    path = Path(model_dir) / "holdout_tokens.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def backtest_pumpfun_model(
    model_dir: str | Path,
    minutes: int = 10,
    test_window: int = 240,
    target_mode: str = "sum",
    max_tokens: int | None = None,
    max_samples: int | None = None,
) -> PumpfunBacktestResult:
    bundle = load_model_artifacts(model_dir)
    config_meta = bundle.metadata.get("config", {})
    model_horizon = int(config_meta.get("horizon", 10))
    context_length = int(config_meta.get("context_length", 336))

    if target_mode == "sum" and minutes > model_horizon:
        raise ValueError(
            f"Requested {minutes} minutes, but model horizon is {model_horizon}. Retrain with a larger horizon."
        )

    db = get_pumpfun_db_manager()
    with db.session() as session:
        holdout_tokens = _load_holdout_tokens(model_dir)
        if not holdout_tokens:
            raise ValueError("No holdout tokens found for backtest")
        if max_tokens is not None:
            holdout_tokens = holdout_tokens[:max_tokens]
        training = prepare_pumpfun_training_data(session, holdout_tokens, normalize=True)

    df = training.df
    if df.empty:
        raise ValueError("No holdout data available for backtest")

    if target_mode == "direct":
        df = add_return_target(df, minutes)
    df = df.dropna().reset_index(drop=True)

    preds_all = []
    actual_all = []
    dir_hits = []
    price_acc = []

    feature_cols = [col for col in [*PUMPFUN_FEATURE_COLUMNS, "log_close", "log_volume"] if col in df.columns]

    for token_id, token_df in df.groupby("token_id"):
        token_df = token_df.sort_values("timestamp").reset_index(drop=True)
        if len(token_df) < context_length + minutes:
            continue
        start_idx = max(context_length, len(token_df) - test_window)
        for idx in range(start_idx, len(token_df) - minutes):
            history = token_df.iloc[idx - context_length : idx].reset_index(drop=True)
            preds = forecast_next_horizon(
                bundle.model,
                history,
                target_col="return_horizon" if target_mode == "direct" else "return",
                feature_cols=feature_cols,
                horizon=model_horizon,
            )
            preds = np.asarray(preds, dtype=float)
            if target_mode == "direct":
                pred_return = float(preds[-1])
                actual_return = float(token_df["return_horizon"].iloc[idx])
            else:
                pred_return = float(np.sum(preds[:minutes]))
                actual_return = float(token_df["return"].iloc[idx : idx + minutes].sum())
            preds_all.append(pred_return)
            actual_all.append(actual_return)
            dir_hits.append(int(np.sign(pred_return) == np.sign(actual_return)))
            if "raw_log_close" in history.columns and "raw_log_close" in token_df.columns:
                last_raw_log_close = float(history["raw_log_close"].iloc[-1])
                actual_raw_log_close = float(token_df["raw_log_close"].iloc[idx + minutes])
                pred_raw_log_close = last_raw_log_close + pred_return
                pred_close = float(np.exp(pred_raw_log_close))
                actual_close = float(np.exp(actual_raw_log_close))
                if actual_close == 0:
                    price_acc.append(float("nan"))
                else:
                    price_acc.append(max(0.0, 1.0 - abs(pred_close - actual_close) / actual_close) * 100.0)
            if max_samples is not None and len(preds_all) >= max_samples:
                break
        if max_samples is not None and len(preds_all) >= max_samples:
            break

    preds_arr = np.asarray(preds_all, dtype=float)
    actual_arr = np.asarray(actual_all, dtype=float)
    if preds_arr.size == 0:
        raise ValueError("No backtest samples generated")

    mae = float(np.mean(np.abs(preds_arr - actual_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - actual_arr) ** 2)))
    denom = np.abs(actual_arr) + np.abs(preds_arr) + 1e-8
    smape = float(np.mean(2.0 * np.abs(preds_arr - actual_arr) / denom)) * 100.0
    direction_accuracy = float(np.mean(dir_hits)) * 100.0
    price_accuracy_pct = float(np.nanmean(price_acc)) if price_acc else float("nan")

    return PumpfunBacktestResult(
        mae=mae,
        rmse=rmse,
        smape=smape,
        direction_accuracy=direction_accuracy,
        price_accuracy_pct=price_accuracy_pct,
        samples=int(preds_arr.size),
    )
