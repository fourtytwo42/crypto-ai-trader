"""Walk-forward backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest.metrics_calculator import (
    calculate_metrics,
    calculate_prediction_metrics,
)
from src.backtest.signal_executor import execute_signals
from src.prediction.predictor import generate_predictions
from src.prediction.signal_generator import apply_signals
from src.data.feature_extractor import get_feature_columns
from src.training.config import TrainingConfig
from src.training.data_preparation import walk_forward_splits
from src.training.trainer import train_model


@dataclass
class BacktestResult:
    """Result from walk-forward backtest."""

    window_metrics: list[dict[str, float]]
    aggregated: dict[str, float]


def walk_forward_backtest(
    data: pd.DataFrame,
    config: TrainingConfig,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    model_dir: str,
    threshold: float = 0.005,
    purge: int = 1,
    force_simple: bool = False,
) -> BacktestResult:
    """Run walk-forward backtest with rolling windows."""
    splits = walk_forward_splits(
        data,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=step_size,
        purge=purge,
    )

    window_metrics: list[dict[str, float]] = []
    for split in splits:
        feature_cols = [col for col in get_feature_columns() if col in split.train.columns]
        result = train_model(
            config,
            split.train,
            split.val,
            model_dir=model_dir,
            target_col="return",
            force_simple=force_simple,
            feature_cols=feature_cols,
        )
        preds = generate_predictions(result.model, split.test, feature_cols=feature_cols)
        signals = apply_signals(
            [{"q10": p["q10"], "q50": p["q50"], "q90": p["q90"]} for p in preds],
            threshold=threshold,
        )
        aligned_test = split.test.tail(len(signals)).reset_index(drop=True)
        trades = execute_signals(aligned_test, signals)
        metrics = calculate_metrics(trades)

        target_col = "target_return" if "target_return" in split.test.columns else "return"
        y_true = aligned_test[target_col].to_numpy(dtype=float)
        y_pred = [float(item["q50"]) for item in preds]
        if len(y_pred) < len(y_true):
            y_true = y_true[-len(y_pred):]
        pred_metrics = calculate_prediction_metrics(list(y_true), y_pred)
        metrics.update(pred_metrics)

        window_metrics.append(metrics)

    if not window_metrics:
        aggregated = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "turnover": 0.0,
            "directional_accuracy": 0.0,
            "max_correct_streak": 0.0,
            "avg_correct_streak": 0.0,
        }
    else:
        keys = window_metrics[0].keys()
        aggregated = {k: float(sum(m[k] for m in window_metrics) / len(window_metrics)) for k in keys}

    return BacktestResult(window_metrics=window_metrics, aggregated=aggregated)
