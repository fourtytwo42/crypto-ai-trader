"""Walk-forward backtesting with volatility-adaptive signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

from src.backtest.metrics_calculator import (
    calculate_metrics,
    calculate_prediction_metrics,
)
from src.backtest.signal_executor import execute_signals
from src.data.feature_extractor import calculate_baseline_volatility, get_feature_columns
from src.prediction.predictor import generate_predictions
from src.prediction.signal_generator import SignalConfig, apply_signals
from src.training.config import TrainingConfig
from src.training.data_preparation import walk_forward_splits
from src.training.trainer import train_model
from src.training.evaluator import calculate_metrics as calculate_forecast_metrics

logger = structlog.get_logger(__name__)

@dataclass
class BacktestResult:
    """Result from walk-forward backtest."""

    window_metrics: list[dict[str, float]]
    aggregated: dict[str, float]


@dataclass
class BacktestConfig:
    """Configuration for backtesting with enhanced signal generation."""

    threshold: float = 0.005
    volatility_adaptive: bool = True
    baseline_volatility: float | None = None  # Auto-calculate if None
    volatility_window: str = "ret_std_7"  # Column name for current volatility
    max_uncertainty_spread: float = 0.03
    uncertainty_enabled: bool = True
    min_holding_periods: int = 1


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
    backtest_config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run walk-forward backtest with rolling windows.

    Enhanced with volatility-adaptive signals and uncertainty filtering.

    Args:
        data: Full dataset with features
        config: Training configuration
        train_size: Training window size in periods
        val_size: Validation window size in periods
        test_size: Test window size in periods
        step_size: Step size between windows
        model_dir: Directory for model artifacts
        threshold: Base signal threshold (used if backtest_config not provided)
        purge: Gap between train and validation
        force_simple: Use simple model for testing
        backtest_config: Enhanced backtest configuration with volatility settings

    Returns:
        BacktestResult with window metrics and aggregated metrics
    """
    # Initialize backtest config if not provided
    if backtest_config is None:
        backtest_config = BacktestConfig(threshold=threshold)

    # Calculate baseline volatility from initial training data if not specified
    if backtest_config.volatility_adaptive and backtest_config.baseline_volatility is None:
        train_subset = data.iloc[:train_size]
        if "return" in train_subset.columns:
            backtest_config.baseline_volatility = calculate_baseline_volatility(
                train_subset, column="return"
            )

    # Build signal config from backtest config
    signal_config = SignalConfig(
        base_threshold=backtest_config.threshold,
        volatility_adaptive=backtest_config.volatility_adaptive,
        baseline_volatility=backtest_config.baseline_volatility or 0.02,
        uncertainty_enabled=backtest_config.uncertainty_enabled,
        max_uncertainty_spread=backtest_config.max_uncertainty_spread,
    )

    splits = walk_forward_splits(
        data,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=step_size,
        purge=purge,
    )

    window_metrics: list[dict[str, float]] = []
    running_mae = 0.0
    running_rmse = 0.0
    running_mape = 0.0
    running_count = 0
    total_splits = len(splits)
    for idx, split in enumerate(splits, start=1):
        logger.info("Backtest window start", window=idx, total_windows=total_splits)
        feature_cols = [col for col in get_feature_columns() if col in split.train.columns]
        target_col = "target_return" if "target_return" in split.train.columns else "return"
        result = train_model(
            config,
            split.train,
            split.val,
            model_dir=model_dir,
            target_col=target_col,
            force_simple=force_simple,
            feature_cols=feature_cols,
            save_artifacts=False,
        )
        preds = generate_predictions(
            result.model,
            split.test,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        # Align test data with predictions
        aligned_test = split.test.tail(len(preds)).reset_index(drop=True)

        # Extract volatility series for adaptive threshold
        volatility_series: list[float] | None = None
        vol_col = backtest_config.volatility_window
        if backtest_config.volatility_adaptive and vol_col in aligned_test.columns:
            volatility_series = aligned_test[vol_col].tolist()

        # Generate signals with enhanced config
        signals = apply_signals(
            [{"q10": p["q10"], "q50": p["q50"], "q90": p["q90"]} for p in preds],
            threshold=backtest_config.threshold,
            volatility_series=volatility_series,
            config=signal_config,
        )

        metrics: dict[str, float] = {}
        trades = execute_signals(aligned_test, signals)
        if target_col == "return":
            metrics = calculate_metrics(trades)

        y_true = aligned_test[target_col].to_numpy(dtype=float)
        y_pred = [float(item["q50"]) for item in preds]
        if len(y_pred) < len(y_true):
            y_true = y_true[-len(y_pred) :]
        pred_metrics = calculate_prediction_metrics(list(y_true), y_pred)
        forecast_metrics = calculate_forecast_metrics(
            y_true,
            pd.Series(y_pred, dtype=float).to_numpy(),
        )
        metrics.update(pred_metrics)
        metrics.update(forecast_metrics)

        window_metrics.append(metrics)
        running_count += 1
        running_mae += float(metrics.get("mae", 0.0))
        running_rmse += float(metrics.get("rmse", 0.0))
        running_mape += float(metrics.get("mape", 0.0))
        avg_mae = running_mae / running_count
        avg_rmse = running_rmse / running_count
        avg_mape = running_mape / running_count
        logger.info(
            "Backtest window end",
            window=idx,
            total_windows=total_splits,
            metrics=metrics,
            avg_mae=avg_mae,
            avg_rmse=avg_rmse,
            avg_mape=avg_mape,
        )

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
        aggregated = {
            k: float(sum(m[k] for m in window_metrics) / len(window_metrics)) for k in keys
        }

    return BacktestResult(window_metrics=window_metrics, aggregated=aggregated)
