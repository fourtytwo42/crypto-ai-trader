"""Walk-forward forecasting for multi-step candle prediction."""

from __future__ import annotations

from dataclasses import dataclass
import gc

import numpy as np
import pandas as pd
import structlog
import torch

from src.training.config import TrainingConfig
from src.training.data_preparation import to_neuralforecast_format, walk_forward_splits
from src.training.evaluator import calculate_metrics
from src.training.model_factory import SimpleQuantileModel, create_model
from src.training.progress import ProgressDisplay, set_active_display

logger = structlog.get_logger(__name__)

FORECAST_FEATURES = [
    "log_close",
    "log_volume",
    "return",
    "range",
    "body",
    "dlog_volume",
    "ret_mean_7",
    "ret_std_7",
    "ret_mean_30",
    "ret_std_30",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


@dataclass
class ForecastResult:
    """Result from walk-forward forecasting."""

    window_metrics: list[dict[str, float]]
    aggregated: dict[str, float]


def _select_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    return [col for col in FORECAST_FEATURES if col in df.columns and col != target_col]


def _log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    logger.info(
        "CUDA memory",
        tag=tag,
        allocated_mb=round(allocated, 1),
        reserved_mb=round(reserved, 1),
        max_allocated_mb=round(max_allocated, 1),
    )


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _extract_forecast_values(forecast: pd.DataFrame) -> np.ndarray:
    columns = list(forecast.columns)
    if "y" in columns:
        values = forecast["y"].to_numpy(dtype=float)
    else:
        value_cols = [col for col in columns if col not in {"ds", "unique_id"}]
        if not value_cols:
            raise ValueError("forecast output missing prediction columns")
        values = forecast[value_cols[0]].to_numpy(dtype=float)
    return values


def _extract_forecast_values_multi(forecast: pd.DataFrame) -> dict[str, np.ndarray]:
    columns = list(forecast.columns)
    if "y" in columns:
        value_col = "y"
    else:
        value_cols = [col for col in columns if col not in {"ds", "unique_id"}]
        if not value_cols:
            raise ValueError("forecast output missing prediction columns")
        value_col = value_cols[0]
    results: dict[str, np.ndarray] = {}
    for uid, group in forecast.groupby("unique_id"):
        values = group[value_col].to_numpy(dtype=float)
        results[str(uid)] = values
    return results


def _fit_and_forecast(
    history_df: pd.DataFrame,
    target_col: str,
    config: TrainingConfig,
    feature_cols: list[str],
    unique_id_col: str | None = None,
) -> np.ndarray:
    model = create_model(config, force_simple=False, feature_cols=feature_cols)
    if isinstance(model, SimpleQuantileModel):
        y_history = history_df[target_col].to_numpy(dtype=float)
        model.fit(y_history)
        return model.predict(config.horizon)["q50"]

    history_nf = to_neuralforecast_format(
        history_df,
        target_col=target_col,
        feature_cols=feature_cols,
        unique_id_col=unique_id_col,
    )
    _log_cuda_memory("before_fit")
    model.fit(history_nf)
    _log_cuda_memory("after_fit")
    forecast = model.predict(history_nf)
    _log_cuda_memory("after_predict")
    values = _extract_forecast_values(forecast)
    if values.size > config.horizon:
        values = values[-config.horizon:]
    _cleanup_cuda()
    return values


def _fit_and_forecast_multi(
    history_df: pd.DataFrame,
    target_col: str,
    config: TrainingConfig,
    feature_cols: list[str],
    unique_id_col: str,
) -> dict[str, np.ndarray]:
    model = create_model(config, force_simple=False, feature_cols=feature_cols)
    if isinstance(model, SimpleQuantileModel):
        results: dict[str, np.ndarray] = {}
        for uid, group in history_df.groupby(unique_id_col):
            y_history = group[target_col].to_numpy(dtype=float)
            simple = SimpleQuantileModel(quantiles=config.quantiles)
            simple.fit(y_history)
            results[str(uid)] = simple.predict(config.horizon)["q50"]
        return results

    history_nf = to_neuralforecast_format(
        history_df,
        target_col=target_col,
        feature_cols=feature_cols,
        unique_id_col=unique_id_col,
    )
    _log_cuda_memory("before_fit")
    model.fit(history_nf)
    _log_cuda_memory("after_fit")
    forecast = model.predict(history_nf)
    _log_cuda_memory("after_predict")
    values_by_id = _extract_forecast_values_multi(forecast)
    trimmed: dict[str, np.ndarray] = {}
    for uid, values in values_by_id.items():
        if values.size > config.horizon:
            values = values[-config.horizon:]
        trimmed[uid] = values
    _cleanup_cuda()
    return trimmed


def _directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_observed: float,
) -> float:
    prev_actual = np.concatenate(([last_observed], y_true[:-1]))
    actual_direction = np.sign(y_true - prev_actual)
    pred_direction = np.sign(y_pred - prev_actual)
    return float(np.mean(actual_direction == pred_direction))


def _within_tolerance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float,
) -> float:
    if np.any(y_true == 0):
        return float(np.nan)
    return float(np.mean(np.abs((y_pred - y_true) / y_true) <= tolerance))


def _per_horizon_close_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    last_observed: float,
    tolerances: tuple[float, ...] = (0.005, 0.01),
) -> dict[str, float]:
    prev_actual = np.concatenate(([last_observed], y_true[:-1]))
    direction_ok = np.sign(y_true - prev_actual) == np.sign(y_pred - prev_actual)
    results: dict[str, float] = {}
    for idx, ok in enumerate(direction_ok, start=1):
        results[f"close_directional_accuracy_h{idx}"] = float(ok)
    for tol in tolerances:
        if np.any(y_true == 0):
            within = np.full_like(y_true, np.nan, dtype=float)
        else:
            within = (np.abs((y_pred - y_true) / y_true) <= tol).astype(float)
        pct = int(tol * 1000) / 10
        key_prefix = f"close_within_{pct}pct_h"
        for idx, ok in enumerate(within, start=1):
            results[f"{key_prefix}{idx}"] = float(ok)
    return results


def _multi_asset_splits(
    data: pd.DataFrame,
    symbol_col: str,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
) -> list[dict[str, "DataSplits"]]:
    if symbol_col not in data.columns:
        raise ValueError("multi-asset requires symbol column")
    splits_by_symbol: dict[str, list] = {}
    for symbol, symbol_df in data.groupby(symbol_col):
        symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True)
        splits = walk_forward_splits(
            symbol_df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
        )
        if splits:
            splits_by_symbol[str(symbol)] = splits
    if not splits_by_symbol:
        raise ValueError("no walk-forward splits available for any symbol")
    min_splits = min(len(splits) for splits in splits_by_symbol.values())
    return [
        {symbol: splits_by_symbol[symbol][idx] for symbol in splits_by_symbol}
        for idx in range(min_splits)
    ]


def _evaluate_forecast(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    label: str,
    last_observed: float | None = None,
) -> dict[str, float]:
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    metrics = calculate_metrics(y_true, y_pred)
    results = {f"{label}_{key}": float(value) for key, value in metrics.items()}
    if label == "close" and last_observed is not None:
        results["close_directional_accuracy"] = _directional_accuracy(
            y_true,
            y_pred,
            last_observed=last_observed,
        )
        results["close_within_0_5pct"] = _within_tolerance(y_true, y_pred, tolerance=0.005)
        results["close_within_1pct"] = _within_tolerance(y_true, y_pred, tolerance=0.01)
        results.update(
            _per_horizon_close_metrics(
                y_true,
                y_pred,
                last_observed=last_observed,
            )
        )
    return results


def walk_forward_forecast(
    data: pd.DataFrame,
    config: TrainingConfig,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    close_target_col: str = "log_close",
    start_window: int | None = None,
    max_windows: int | None = None,
    multi_asset: bool = False,
    symbol_col: str = "symbol",
) -> ForecastResult:
    """Run walk-forward forecast for close and volume.

    Uses log-close and log-volume series with multi-step horizon predictions.
    """
    if multi_asset:
        multi_splits = _multi_asset_splits(
            data,
            symbol_col=symbol_col,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
        )
        total_splits = len(multi_splits)
    else:
        splits = walk_forward_splits(
            data,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            step_size=step_size,
        )

        if not splits:
            raise ValueError("no walk-forward splits available")

        total_splits = len(splits)
    start_idx = 1 if start_window is None else max(1, start_window)
    if start_idx > total_splits:
        raise ValueError("start_window exceeds available windows")
    end_idx = total_splits
    if max_windows is not None:
        if max_windows <= 0:
            raise ValueError("max_windows must be positive")
        end_idx = min(total_splits, start_idx - 1 + max_windows)
    if multi_asset:
        splits_subset = multi_splits[start_idx - 1 : end_idx]
    else:
        splits_subset = splits[start_idx - 1 : end_idx]

    window_metrics: list[dict[str, float]] = []
    display = ProgressDisplay(
        total_windows=total_splits,
        total_epochs=config.epochs,
        window_label="Forecast windows",
        epoch_label="Training epochs",
    )
    set_active_display(display)
    try:
        for idx, split in enumerate(splits_subset, start=start_idx):
            display.set_window(idx, total=total_splits)
            _log_cuda_memory("window_start")
            metrics: dict[str, float] = {}
            if multi_asset:
                history_df = pd.concat(
                    [pd.concat([s.train, s.val], ignore_index=True) for s in split.values()],
                    ignore_index=True,
                )
                for target_col, label in (("log_close", "close"), ("log_volume", "volume")):
                    if label == "close":
                        target_col = close_target_col
                    if target_col not in history_df.columns:
                        raise ValueError(f"missing required column: {target_col}")
                    feature_cols = _select_feature_cols(history_df, target_col)
                    preds_by_symbol = _fit_and_forecast_multi(
                        history_df,
                        target_col,
                        config,
                        feature_cols,
                        unique_id_col=symbol_col,
                    )
                    per_symbol_metrics: list[dict[str, float]] = []
                    for symbol, symbol_split in split.items():
                        symbol_history = pd.concat(
                            [symbol_split.train, symbol_split.val], ignore_index=True
                        )
                        test_df = symbol_split.test.head(config.horizon).reset_index(drop=True)
                        if len(test_df) < config.horizon:
                            continue
                        last_observed_close = float(np.exp(symbol_history["log_close"].iloc[-1]))
                        last_log_close = float(symbol_history["log_close"].iloc[-1])
                        preds = preds_by_symbol.get(str(symbol))
                        if preds is None:
                            continue
                        y_true = test_df[target_col].to_numpy(dtype=float)
                        min_len = min(len(preds), len(y_true))
                        if min_len == 0:
                            continue
                        preds = preds[:min_len]
                        y_true = y_true[:min_len]
                        if label == "close" and target_col == "return":
                            y_true_log = last_log_close + np.cumsum(y_true)
                            y_pred_log = last_log_close + np.cumsum(preds)
                            per_symbol_metrics.append(
                                _evaluate_forecast(
                                    y_true_log,
                                    y_pred_log,
                                    label,
                                    last_observed=last_observed_close,
                                )
                            )
                        else:
                            per_symbol_metrics.append(
                                _evaluate_forecast(
                                    y_true,
                                    preds,
                                    label,
                                    last_observed=last_observed_close if label == "close" else None,
                                )
                            )
                    if per_symbol_metrics:
                        keys = per_symbol_metrics[0].keys()
                        metrics.update(
                            {
                                k: float(sum(m[k] for m in per_symbol_metrics) / len(per_symbol_metrics))
                                for k in keys
                            }
                        )
            else:
                history_df = pd.concat([split.train, split.val], ignore_index=True)
                test_df = split.test.head(config.horizon).reset_index(drop=True)
                if len(test_df) < config.horizon:
                    raise ValueError("insufficient test data for forecast horizon")

                last_observed_close = float(np.exp(history_df["log_close"].iloc[-1]))
                last_log_close = float(history_df["log_close"].iloc[-1])
                for target_col, label in (("log_close", "close"), ("log_volume", "volume")):
                    if label == "close":
                        target_col = close_target_col
                    if target_col not in history_df.columns:
                        raise ValueError(f"missing required column: {target_col}")
                    feature_cols = _select_feature_cols(history_df, target_col)
                    preds = _fit_and_forecast(history_df, target_col, config, feature_cols)
                    y_true = test_df[target_col].to_numpy(dtype=float)
                    min_len = min(len(preds), len(y_true))
                    if min_len == 0:
                        raise ValueError("forecast length is zero")
                    preds = preds[:min_len]
                    y_true = y_true[:min_len]
                    if label == "close" and target_col == "return":
                        y_true_log = last_log_close + np.cumsum(y_true)
                        y_pred_log = last_log_close + np.cumsum(preds)
                        metrics.update(
                            _evaluate_forecast(
                                y_true_log,
                                y_pred_log,
                                label,
                                last_observed=last_observed_close,
                            )
                        )
                    else:
                        metrics.update(
                            _evaluate_forecast(
                                y_true,
                                preds,
                                label,
                                last_observed=last_observed_close if label == "close" else None,
                            )
                        )

            window_metrics.append(metrics)
            logger.info("Forecast window end", window=idx, metrics=metrics)
            display.finish_window()
            _cleanup_cuda()
    finally:
        set_active_display(None)

    keys = window_metrics[0].keys()
    aggregated = {
        k: float(sum(m[k] for m in window_metrics) / len(window_metrics)) for k in keys
    }
    return ForecastResult(window_metrics=window_metrics, aggregated=aggregated)
