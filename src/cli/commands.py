"""CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd
import structlog
import numpy as np
import json

from src.data.feature_extractor import extract_features_hourly, get_feature_columns
from src.data.kucoin_client import backfill_kucoin_candles
from src.data.normalizer import RollingNormalizer
from src.forecasting.recent_features import (
    build_recent_feature_df_from_candles,
    fetch_recent_hourly_candles,
)
from src.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from src.config import get_settings
from src.data.pipeline import DataPipeline, prepare_training_data, prepare_training_data_multi
from src.database.connection import get_db_manager
from src.database.operations import get_latest_features
from src.prediction.model_loader import load_model_artifacts
from src.prediction.predictor import generate_predictions
from src.prediction.signal_generator import SignalConfig, generate_signal
from src.forecasting.walk_forward_forecast import (
    FORECAST_FEATURES,
    walk_forward_forecast,
)
from src.forecasting.predict import build_future_timestamps, forecast_next_horizon
from src.training.config import TrainingConfig
from src.training.trainer import train_model
from src.pumpfun.backtest import backtest_pumpfun_model
from src.pumpfun.classifier import (
    backtest_pumpfun_direction_classifier,
    train_pumpfun_direction_classifier,
)
from src.pumpfun.db import get_pumpfun_db_manager
from src.pumpfun.pipeline import PumpfunFilterConfig, sync_pumpfun_candles
from src.pumpfun.predict import predict_pumpfun
from src.pumpfun.training import train_pumpfun_model

logger = structlog.get_logger(__name__)


def _build_horizon_weights(mode: str, horizon: int) -> list[float] | None:
    if mode == "none":
        return None
    if horizon <= 0:
        raise ValueError("horizon must be positive for horizon weighting")
    if mode == "linear":
        weights = np.linspace(1.0, 2.0, horizon)
    elif mode == "quadratic":
        weights = np.linspace(1.0, 2.0, horizon) ** 2
    elif mode == "cubic":
        weights = np.linspace(1.0, 2.0, horizon) ** 3
    elif mode == "exp":
        weights = np.geomspace(1.0, 4.0, horizon)
    else:
        raise ValueError(f"unsupported horizon_weight_mode: {mode}")
    return weights.tolist()


def _add_return_24h_target(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    df = df.copy()
    if symbol_col in df.columns:
        df = df.sort_values([symbol_col, "timestamp"]).reset_index(drop=True)
        df["return_24h"] = df.groupby(symbol_col)["log_close"].shift(-24) - df["log_close"]
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["return_24h"] = df["log_close"].shift(-24) - df["log_close"]
    return df


def _fetch_recent_hourly_candles(symbol: str, hours_back: int) -> list:
    """Fetch the most recent hourly candles for a symbol."""
    return fetch_recent_hourly_candles(symbol, hours_back)


def _build_recent_feature_df_from_candles(
    candles: list,
    context_length: int,
    normalize_cols: list[str],
) -> tuple[pd.DataFrame, RollingNormalizer | None]:
    """Convert raw candles into normalized feature DataFrame."""
    return build_recent_feature_df_from_candles(candles, context_length, normalize_cols)


def forecast_holdout_24h_command(
    context_length: int,
    hidden_size: int,
    num_layers: int,
    patch_length: int,
    stride: int,
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_type: str,
    nhits_stack_types: list[str] | None = None,
    nhits_n_blocks: list[int] | None = None,
    nhits_mlp_units: list[list[int]] | None = None,
    nhits_n_pool_kernel_size: list[int] | None = None,
    nhits_n_freq_downsample: list[int] | None = None,
    symbols: list[str] | None = None,
    multi_asset: bool = False,
) -> dict[str, object]:
    """Train on data older than last 24 hours and evaluate on latest 24 hours."""
    settings = get_settings()
    config = TrainingConfig(
        model_type=model_type,
        horizon=1,
        device=settings.train_device,
        data_frequency="1hour",
        context_length=context_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        loss_type=loss_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        nhits_stack_types=nhits_stack_types,
        nhits_n_blocks=nhits_n_blocks,
        nhits_mlp_units=nhits_mlp_units,
        nhits_n_pool_kernel_size=nhits_n_pool_kernel_size,
        nhits_n_freq_downsample=nhits_n_freq_downsample,
    )

    db = get_db_manager()
    with db.session() as session:
        normalize_cols = get_feature_columns()
        if "return" in normalize_cols:
            normalize_cols = [col for col in normalize_cols if col != "return"]
        normalize_cols.append("log_close")
        normalize_cols = list(dict.fromkeys(normalize_cols))
        features_df, _ = prepare_training_data_multi(
            session,
            symbols=symbols,
            normalize=True,
            normalize_cols=normalize_cols,
        )
        multi_asset = True

    if features_df.empty:
        raise ValueError("no features available for 24h holdout")

    features_df = features_df.dropna().reset_index(drop=True)
    features_df = _add_return_24h_target(features_df)
    features_df = features_df.dropna().reset_index(drop=True)

    symbol_col = "symbol"
    train_frames: list[pd.DataFrame] = []
    eval_items: list[dict[str, int | str]] = []

    for symbol, symbol_df in features_df.groupby(symbol_col):
        symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True)
        if len(symbol_df) < context_length + 48:
            continue
        train_end = len(symbol_df) - 48
        train_frames.append(symbol_df.iloc[:train_end].reset_index(drop=True))
        eval_start = len(symbol_df) - 48
        eval_end = len(symbol_df) - 24
        for idx in range(eval_start, eval_end):
            eval_items.append(
                {
                    "symbol": str(symbol),
                    "input_idx": idx,
                    "target_idx": idx + 24,
                }
            )

    if not train_frames or not eval_items:
        raise ValueError("insufficient data for 24h holdout evaluation")

    train_df = pd.concat(train_frames, ignore_index=True)
    val_size = max(int(len(train_df) * settings.train_validation_split), 30)
    train_split = train_df.iloc[:-val_size].reset_index(drop=True)
    val_split = train_df.iloc[-val_size:].reset_index(drop=True)

    feature_cols = [col for col in FORECAST_FEATURES if col in train_df.columns]

    result = train_model(
        config,
        train_split,
        val_split,
        model_dir="models_holdout_24h_tmp",
        target_col="return_24h",
        force_simple=False,
        feature_cols=feature_cols,
        save_artifacts=False,
        keep_best=False,
        unique_id_col=symbol_col if multi_asset else None,
    )

    by_symbol: dict[str, list[dict[str, float]]] = {}
    for item in eval_items:
        symbol = str(item["symbol"])
        symbol_df = features_df[features_df[symbol_col] == symbol].reset_index(drop=True)
        input_idx = int(item["input_idx"])
        target_idx = int(item["target_idx"])
        history_df = symbol_df.iloc[: input_idx + 1].tail(context_length).reset_index(drop=True)
        if len(history_df) < context_length:
            continue
        last_log_close = float(history_df["log_close"].iloc[-1])
        preds = forecast_next_horizon(
            result.model,
            history_df,
            target_col="return_24h",
            feature_cols=feature_cols,
            horizon=1,
        )
        pred_return = float(np.asarray(preds, dtype=float)[-1])
        pred_log_close = last_log_close + pred_return
        pred_close = float(np.exp(pred_log_close))
        actual_log_close = float(symbol_df["log_close"].iloc[target_idx])
        actual_close = float(np.exp(actual_log_close))
        prev_close = float(np.exp(last_log_close))
        actual_dir = float(np.sign(actual_close - prev_close))
        pred_dir = float(np.sign(pred_close - prev_close))
        dir_ok = float(actual_dir == pred_dir)
        if actual_close == 0:
            acc_pct = float("nan")
        else:
            acc_pct = max(0.0, 1.0 - abs(pred_close - actual_close) / actual_close) * 100.0
        by_symbol.setdefault(symbol, []).append(
            {
                "pred_close": pred_close,
                "actual_close": actual_close,
                "direction_ok": dir_ok,
                "accuracy_pct": acc_pct,
            }
        )

    summary: dict[str, dict[str, float]] = {}
    for symbol, rows in by_symbol.items():
        if not rows:
            continue
        direction = float(np.mean([r["direction_ok"] for r in rows])) * 100.0
        accuracy = float(np.nanmean([r["accuracy_pct"] for r in rows]))
        summary[symbol] = {
            "directional_accuracy_pct": direction,
            "price_accuracy_pct": accuracy,
            "samples": float(len(rows)),
        }

    return {
        "summary": summary,
        "train_metrics": result.metrics,
    }


def load_data_command(
    file_path: str,
    symbol: str,
    replace_existing: bool = False,
) -> dict[str, int]:
    """Load CSV data into the database."""
    db = get_db_manager()
    with db.session() as session:
        pipeline = DataPipeline(session)
        return pipeline.run_full_pipeline(file_path, symbol, replace_existing=replace_existing)


def train_command(model_dir: str | None = None) -> dict[str, float]:
    """Train a model using features in the database."""
    settings = get_settings()
    model_dir = model_dir or settings.model_dir

    db = get_db_manager()
    with db.session() as session:
        features_df, _ = prepare_training_data(session)

    if features_df.empty:
        raise ValueError("no features available for training")

    features_df = features_df.dropna().reset_index(drop=True)
    if features_df.empty:
        raise ValueError("no clean rows available after dropping missing values")

    config = TrainingConfig(model_type="patchtst", device=settings.train_device)
    splits = features_df.iloc[:-10], features_df.iloc[-10:]
    result = train_model(
        config,
        splits[0],
        splits[1],
        model_dir=model_dir,
        target_col="return",
        force_simple=False,
        keep_best=True,
    )
    return result.metrics


def predict_command(
    model_dir: str,
    data: pd.DataFrame | None = None,
    volatility_adaptive: bool = True,
    current_volatility: float | None = None,
    max_uncertainty_spread: float = 0.03,
) -> dict[str, object]:
    """Generate a prediction using latest DB features or provided data.

    Args:
        model_dir: Path to model directory.
        data: Optional DataFrame with features.
        volatility_adaptive: Enable volatility-adaptive threshold.
        current_volatility: Current volatility for adaptive threshold.
        max_uncertainty_spread: Maximum q90-q10 spread before going flat.

    Returns:
        Dictionary with prediction and signal.
    """
    settings = get_settings()
    bundle = load_model_artifacts(model_dir)

    if data is None:
        db = get_db_manager()
        with db.session() as session:
            features = get_latest_features(session, limit=1, symbol=settings.data_symbol)
            if not features:
                raise ValueError("no features available for prediction")
            feature = features[0]
            data = pd.DataFrame(
                [
                    {
                        "timestamp": feature.timestamp,
                        "return": float(feature.return_ or 0.0),
                    }
                ]
            )
            # Use ret_std_7 as current volatility if available
            if current_volatility is None and feature.ret_std_7:
                current_volatility = float(feature.ret_std_7)

    predictions = generate_predictions(bundle.model, data)
    latest = predictions[-1]

    # Build signal config with enhanced options
    signal_config = SignalConfig(
        base_threshold=settings.signal_threshold,
        volatility_adaptive=volatility_adaptive,
        baseline_volatility=settings.signal_baseline_volatility,
        uncertainty_enabled=True,
        max_uncertainty_spread=max_uncertainty_spread,
    )

    prediction_dict = {"q10": latest["q10"], "q50": latest["q50"], "q90": latest["q90"]}
    signal = generate_signal(
        prediction_dict,
        current_volatility=current_volatility,
        config=signal_config,
    )

    return {
        "prediction": latest,
        "signal": signal,
        "volatility_adaptive": volatility_adaptive,
        "current_volatility": current_volatility,
    }


def backtest_command(
    model_dir: str,
    data: pd.DataFrame | None = None,
    volatility_adaptive: bool = True,
    uncertainty_enabled: bool = True,
    max_uncertainty_spread: float = 0.03,
    min_holding_periods: int = 1,
    horizon: int = 12,
    start_window: int | None = None,
    max_windows: int | None = None,
) -> dict[str, float]:
    """Run a walk-forward backtest with enhanced signal generation.

    Args:
        model_dir: Path to model directory.
        data: Optional DataFrame with features.
        volatility_adaptive: Enable volatility-adaptive threshold.
        uncertainty_enabled: Enable uncertainty-based no-trade band.
        max_uncertainty_spread: Maximum q90-q10 spread before going flat.
        min_holding_periods: Minimum periods to hold position.
        horizon: Prediction horizon in periods.

    Returns:
        Dictionary with aggregated backtest metrics.
    """
    settings = get_settings()
    config = TrainingConfig(model_type="patchtst", horizon=horizon)

    if data is None:
        db = get_db_manager()
        with db.session() as session:
            features_df, _ = prepare_training_data(session, normalize=False)
        if features_df.empty:
            raise ValueError("no features available for backtest")
        data = features_df

    data = data.dropna().reset_index(drop=True)
    if horizon > 1:
        data = data.copy()
        data["target_return"] = data["return"].shift(-horizon)
        data = data.dropna().reset_index(drop=True)

    # Build backtest config with enhanced options
    backtest_config = BacktestConfig(
        threshold=settings.signal_threshold,
        volatility_adaptive=volatility_adaptive,
        volatility_window=settings.signal_volatility_window,
        max_uncertainty_spread=max_uncertainty_spread,
        uncertainty_enabled=uncertainty_enabled,
        min_holding_periods=min_holding_periods,
    )

    train_size = settings.backtest_train_window
    test_size = settings.backtest_test_window
    val_size = max(int(train_size * settings.train_validation_split), 30)
    step_size = max(test_size // 2, 30)

    result = walk_forward_backtest(
        data,
        config,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=step_size,
        model_dir=model_dir,
        backtest_config=backtest_config,
        purge=settings.walk_forward_purge_days,
        start_window=start_window,
        max_windows=max_windows,
    )
    return result.aggregated


def forecast_backtest_command(
    data: pd.DataFrame | None = None,
    horizon: int = 12,
    start_window: int | None = None,
    max_windows: int | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    step_size: int | None = None,
    model_type: str = "nhits",
    context_length: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    patch_length: int | None = None,
    stride: int | None = None,
    horizon_weight_mode: str = "none",
    loss_type: str = "mae",
    close_target: str = "log_close",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    symbols: list[str] | None = None,
    multi_asset: bool = False,
) -> dict[str, float]:
    """Run walk-forward forecasting for next-12 hourly candles and volume."""
    settings = get_settings()
    default_context = settings.model_context_length
    default_epochs = settings.train_epochs
    default_batch = settings.train_batch_size
    default_lr = settings.train_learning_rate
    config = TrainingConfig(
        model_type=model_type,
        horizon=horizon,
        device=settings.train_device,
        data_frequency="1hour",
        context_length=context_length or default_context,
        hidden_size=hidden_size or TrainingConfig.hidden_size,
        num_layers=num_layers or TrainingConfig.num_layers,
        patch_length=patch_length or TrainingConfig.patch_length,
        stride=stride or TrainingConfig.stride,
        horizon_weight=_build_horizon_weights(horizon_weight_mode, horizon),
        loss_type=loss_type,
        epochs=epochs or default_epochs,
        batch_size=batch_size or default_batch,
        learning_rate=learning_rate or default_lr,
    )

    if data is None:
        db = get_db_manager()
        with db.session() as session:
            if multi_asset or symbols:
                normalize_cols = get_feature_columns()
                if close_target == "return":
                    normalize_cols = [col for col in normalize_cols if col != "return"]
                    normalize_cols.append("log_close")
                elif close_target in normalize_cols:
                    normalize_cols = [col for col in normalize_cols if col != close_target]
                normalize_cols = list(dict.fromkeys(normalize_cols))
                features_df, _ = prepare_training_data_multi(
                    session,
                    symbols=symbols,
                    normalize=True,
                    normalize_cols=normalize_cols,
                )
                multi_asset = True
            else:
                features_df, _ = prepare_training_data(session, normalize=False)
        if features_df.empty:
            raise ValueError("no features available for forecast backtest")
        data = features_df

    data = data.dropna().reset_index(drop=True)
    if data.empty:
        raise ValueError("no clean rows available for forecast backtest")

    timestamps = pd.to_datetime(data["timestamp"], utc=True).sort_values()
    median_delta = timestamps.diff().median()
    is_hourly = bool(median_delta is not pd.NaT and median_delta <= pd.Timedelta(hours=2))

    default_train = settings.backtest_train_window
    default_test = settings.backtest_test_window
    if is_hourly:
        default_train *= 24
        default_test *= 24
        if context_length is None:
            config.context_length = max(config.context_length, 168)
    train_size = train_size or default_train
    test_size = test_size or default_test
    val_size = val_size or max(int(train_size * settings.train_validation_split), 30)
    step_size = step_size or max(test_size // 2, 30)

    result = walk_forward_forecast(
        data,
        config,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=step_size,
        close_target_col=close_target,
        start_window=start_window,
        max_windows=max_windows,
        multi_asset=multi_asset,
    )
    return result.aggregated


def forecast_train_command(
    model_dir: str,
    horizon: int = 12,
    context_length: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    patch_length: int | None = None,
    stride: int | None = None,
    horizon_weight_mode: str = "none",
    loss_type: str = "mae",
    close_target: str = "log_close",
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    model_type: str = "nhits",
    symbols: list[str] | None = None,
    multi_asset: bool = False,
) -> dict[str, dict[str, float]]:
    """Train and save forecast models for close and volume."""
    settings = get_settings()
    config = TrainingConfig(
        model_type=model_type,
        horizon=horizon,
        device=settings.train_device,
        data_frequency="1hour",
        context_length=context_length or settings.model_context_length,
        hidden_size=hidden_size or TrainingConfig.hidden_size,
        num_layers=num_layers or TrainingConfig.num_layers,
        patch_length=patch_length or TrainingConfig.patch_length,
        stride=stride or TrainingConfig.stride,
        horizon_weight=_build_horizon_weights(horizon_weight_mode, horizon),
        loss_type=loss_type,
        epochs=epochs or settings.train_epochs,
        batch_size=batch_size or settings.train_batch_size,
        learning_rate=learning_rate or settings.train_learning_rate,
    )

    db = get_db_manager()
    with db.session() as session:
        if multi_asset or symbols:
            normalize_cols = get_feature_columns()
            if close_target == "return":
                normalize_cols = [col for col in normalize_cols if col != "return"]
                normalize_cols.append("log_close")
            elif close_target in normalize_cols:
                normalize_cols = [col for col in normalize_cols if col != close_target]
            normalize_cols = list(dict.fromkeys(normalize_cols))
            features_df, _ = prepare_training_data_multi(
                session,
                symbols=symbols,
                normalize=True,
                normalize_cols=normalize_cols,
            )
            multi_asset = True
        else:
            features_df, _ = prepare_training_data(session, normalize=False)

    if features_df.empty:
        raise ValueError("no features available for forecast training")

    features_df = features_df.dropna().reset_index(drop=True)
    if features_df.empty:
        raise ValueError("no clean rows available for forecast training")

    timestamps = pd.to_datetime(features_df["timestamp"], utc=True).sort_values()
    median_delta = timestamps.diff().median()
    is_hourly = bool(median_delta is not pd.NaT and median_delta <= pd.Timedelta(hours=2))
    if is_hourly and context_length is None:
        config.context_length = max(config.context_length, 168)

    val_size = max(int(len(features_df) * settings.train_validation_split), 30)
    train_df = features_df.iloc[:-val_size].reset_index(drop=True)
    val_df = features_df.iloc[-val_size:].reset_index(drop=True)

    results: dict[str, dict[str, float]] = {}
    for target_col, label in (("log_close", "close"), ("log_volume", "volume")):
        if label == "close":
            target_col = close_target
        if target_col not in features_df.columns:
            raise ValueError(f"missing required column: {target_col}")
        feature_cols = [
            col for col in FORECAST_FEATURES if col in features_df.columns and col != target_col
        ]
        result = train_model(
            config,
            train_df,
            val_df,
            model_dir=Path(model_dir) / label,
            target_col=target_col,
            force_simple=False,
            feature_cols=feature_cols,
            save_artifacts=True,
            keep_best=True,
            unique_id_col="symbol" if multi_asset else None,
        )
        results[label] = result.metrics
        metadata_path = Path(model_dir) / label / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text())
            metadata["target_col"] = target_col
            metadata_path.write_text(json.dumps(metadata, indent=2))
        except Exception:
            logger.warning("Failed to update forecast metadata", path=str(metadata_path))

    return results


def forecast_predict_command(
    model_dir: str,
    horizon: int = 12,
) -> dict[str, object]:
    """Generate next-horizon forecasts using saved close/volume models."""
    close_bundle = load_model_artifacts(Path(model_dir) / "close")
    volume_bundle = load_model_artifacts(Path(model_dir) / "volume")

    config_close = close_bundle.metadata.get("config", {})
    close_target = close_bundle.metadata.get("target_col", "log_close")
    context_length = int(config_close.get("context_length", 168))

    db = get_db_manager()
    with db.session() as session:
        features_df, _ = prepare_training_data(session, normalize=False)

    if features_df.empty:
        raise ValueError("no features available for forecast prediction")

    features_df = features_df.dropna().reset_index(drop=True)
    if len(features_df) < context_length:
        raise ValueError("insufficient history for forecast prediction")

    history_df = features_df.tail(context_length).reset_index(drop=True)
    last_ts = history_df["timestamp"].iloc[-1]
    future_timestamps = build_future_timestamps(last_ts, horizon)

    close_features = [
        col
        for col in FORECAST_FEATURES
        if col in history_df.columns and col != close_target
    ]
    volume_features = [
        col for col in FORECAST_FEATURES if col in history_df.columns and col != "log_volume"
    ]

    close_pred = forecast_next_horizon(
        close_bundle.model,
        history_df,
        target_col=close_target,
        feature_cols=close_features,
        horizon=horizon,
    )
    if close_target == "return":
        last_log_close = float(history_df["log_close"].iloc[-1])
        close_log = last_log_close + np.cumsum(close_pred)
    else:
        close_log = close_pred
    volume_log = forecast_next_horizon(
        volume_bundle.model,
        history_df,
        target_col="log_volume",
        feature_cols=volume_features,
        horizon=horizon,
    )

    close_pred = np.exp(np.asarray(close_log, dtype=float))
    volume_pred = np.exp(np.asarray(volume_log, dtype=float))

    forecast = [
        {
            "timestamp": ts,
            "close": float(close_pred[idx]),
            "volume": float(volume_pred[idx]),
        }
        for idx, ts in enumerate(future_timestamps)
    ]

    return {
        "horizon": horizon,
        "last_timestamp": last_ts,
        "forecast": forecast,
    }


def load_hourly_data_command(
    symbol: str = "BTC-USDT",
    hours_back: int = 8760,
    replace_existing: bool = False,
) -> int:
    """Load hourly data from KuCoin for sub-daily predictions.

    Args:
        symbol: KuCoin symbol (default: BTC-USDT).
        hours_back: Number of hours to backfill (default: 8760 = 1 year).
        replace_existing: If True, delete existing candles first.

    Returns:
        Number of candles loaded.
    """
    db = get_db_manager()
    with db.session() as session:
        pipeline = DataPipeline(session)
        return pipeline.load_kucoin_hourly_to_database(
            symbol=symbol,
            hours_back=hours_back,
            replace_existing=replace_existing,
        )


def quick_predict_command(
    hours: int = 24,
    retrain: bool = False,
    symbols: list[str] | None = None,
    model_dir: str = "models_nhits_best",
) -> dict[str, object]:
    """Pull latest context data from KuCoin (no DB refresh) and predict direction."""
    from pathlib import Path

    settings = get_settings()

    if symbols is None:
        symbols = ["BTC-USDT", "ETH-USDT", "LTC-USDT", "XRP-USDT"]

    if retrain:
        logger.info("Retraining NHITS model with best configuration")
        config = TrainingConfig(
            model_type="nhits",
            horizon=1,
            device=settings.train_device,
            data_frequency="1hour",
            context_length=336,
            hidden_size=512,
            num_layers=3,
            patch_length=8,
            stride=4,
            loss_type="huber",
            epochs=50,
            batch_size=16,
            learning_rate=5e-5,
            nhits_stack_types=["identity", "identity", "identity"],
            nhits_n_blocks=[3, 2, 2],
            nhits_mlp_units=[[768, 768], [768, 768], [768, 768]],
            nhits_n_pool_kernel_size=[2, 2, 1],
            nhits_n_freq_downsample=[4, 2, 1],
        )

        db = get_db_manager()
        with db.session() as session:
            normalize_cols = get_feature_columns()
            if "return" in normalize_cols:
                normalize_cols = [col for col in normalize_cols if col != "return"]
            normalize_cols.append("log_close")
            normalize_cols = list(dict.fromkeys(normalize_cols))

            features_df, _ = prepare_training_data_multi(
                session,
                symbols=symbols,
                normalize=True,
                normalize_cols=normalize_cols,
            )

        if features_df.empty:
            raise ValueError("No features available for training")

        features_df = features_df.dropna().reset_index(drop=True)
        features_df = _add_return_24h_target(features_df)
        features_df = features_df.dropna().reset_index(drop=True)

        train_frames = []
        for symbol in symbols:
            symbol_df = features_df[features_df["symbol"] == symbol].copy()
            symbol_df = symbol_df.sort_values("timestamp").reset_index(drop=True)
            if len(symbol_df) > 48:
                train_frames.append(symbol_df.iloc[:-48].reset_index(drop=True))

        if not train_frames:
            raise ValueError("Insufficient data for training")

        train_df = pd.concat(train_frames, ignore_index=True)
        val_size = max(int(len(train_df) * settings.train_validation_split), 30)
        train_split = train_df.iloc[:-val_size].reset_index(drop=True)
        val_split = train_df.iloc[-val_size:].reset_index(drop=True)

        feature_cols = [col for col in FORECAST_FEATURES if col in train_df.columns]

        result = train_model(
            config,
            train_split,
            val_split,
            model_dir=model_dir,
            target_col="return_24h",
            force_simple=False,
            feature_cols=feature_cols,
            save_artifacts=True,
            keep_best=True,
            unique_id_col="symbol",
        )
        logger.info("Training complete", metrics=result.metrics)
        trained_model = result.model
        context_length = config.context_length
    else:
        model_path = Path(model_dir) / "model.pt"
        if not model_path.exists():
            raise ValueError(
                f"No trained model found at {model_dir}. "
                "Run with --retrain to train a new model first."
            )
        bundle = load_model_artifacts(model_dir)
        trained_model = bundle.model
        config_meta = bundle.metadata.get("config", {})
        context_length = int(config_meta.get("context_length", 336))
        logger.info("Loaded existing model", model_dir=model_dir, context_length=context_length)

    normalize_cols = get_feature_columns()
    if "return" in normalize_cols:
        normalize_cols = [col for col in normalize_cols if col != "return"]
    normalize_cols.append("log_close")
    normalize_cols = list(dict.fromkeys(normalize_cols))

    predictions: dict[str, dict[str, object]] = {}
    feature_cols: list[str] = []

    for symbol in symbols:
        required_history = context_length + 24 + 30
        fetch_hours = required_history + 720
        candles = _fetch_recent_hourly_candles(symbol, fetch_hours)
        if not candles:
            logger.warning("No candles available", symbol=symbol)
            continue

        symbol_df, normalizer = _build_recent_feature_df_from_candles(
            candles, required_history, normalize_cols
        )
        if symbol_df.empty:
            logger.warning("No features available for prediction", symbol=symbol)
            continue

        symbol_df = symbol_df.dropna().reset_index(drop=True)
        symbol_df = _add_return_24h_target(symbol_df)
        symbol_df = symbol_df.dropna().reset_index(drop=True)

        if not feature_cols:
            feature_cols = [col for col in FORECAST_FEATURES if col in symbol_df.columns]

        if len(symbol_df) < context_length:
            logger.warning(
                "Insufficient data for prediction",
                symbol=symbol,
                available=len(symbol_df),
                required=context_length,
            )
            continue

        history_df = symbol_df.tail(context_length).reset_index(drop=True)
        last_timestamp = history_df["timestamp"].iloc[-1]
        latest_price = float(candles[-1].close)
        latest_ts = datetime.fromtimestamp(candles[-1].timestamp, tz=timezone.utc).isoformat()

        log_close_std = 1.0
        if normalizer and "log_close" in normalizer._feature_stats:
            log_close_std = normalizer._feature_stats["log_close"].get("std", 1.0)
            logger.debug("Got log_close std", symbol=symbol, std=log_close_std)

        cumulative_normalized_return = 0.0
        num_predictions = max(1, hours // 24)

        for i in range(num_predictions):
            preds = forecast_next_horizon(
                trained_model,
                history_df,
                target_col="return_24h",
                feature_cols=feature_cols,
                horizon=1,
            )
            pred_return = float(np.asarray(preds, dtype=float)[-1])
            cumulative_normalized_return += pred_return

            if i < num_predictions - 1:
                history_df = history_df.iloc[24:].reset_index(drop=True)
                if len(history_df) < context_length:
                    break

        actual_log_return = cumulative_normalized_return * log_close_std
        price_change_ratio = float(np.exp(actual_log_return))
        price_change_pct = (price_change_ratio - 1) * 100
        predicted_price = latest_price * price_change_ratio
        price_change = predicted_price - latest_price

        if price_change_pct > 0:
            direction = "UP"
        elif price_change_pct < 0:
            direction = "DOWN"
        else:
            direction = "FLAT"

        predictions[symbol] = {
            "current_price": round(latest_price, 2),
            "predicted_price": round(predicted_price, 2),
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "direction": direction,
            "prediction_hours": hours,
            "last_data_timestamp": latest_ts,
        }

    return {
        "predictions": predictions,
        "retrained": retrain,
        "model_dir": model_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def menu_command() -> None:
    """Launch interactive menu."""
    from src.cli.menu import run_menu

    run_menu()


def pumpfun_sync_command(
    min_age_minutes: int = 60,
    active_age_minutes: int = 240,
    min_total_trades: int = 10,
    min_recent_trades: int = 10,
    recent_window_minutes: int = 30,
    replace_existing: bool = False,
    max_tokens: int | None = None,
    price_lookup_enabled: bool = True,
) -> dict[str, int]:
    """Sync pump.fun trades into minute candles/features."""
    config = PumpfunFilterConfig(
        min_age_minutes=min_age_minutes,
        active_age_minutes=active_age_minutes,
        min_total_trades=min_total_trades,
        min_recent_trades=min_recent_trades,
        recent_window_minutes=recent_window_minutes,
    )
    db = get_pumpfun_db_manager()
    with db.session() as session:
        return sync_pumpfun_candles(
            session,
            config,
            replace_existing=replace_existing,
            max_tokens=max_tokens,
            price_lookup_enabled=price_lookup_enabled,
        )


def pumpfun_train_command(
    model_dir: str,
    horizon_minutes: int = 10,
    context_length: int = 336,
    model_type: str = "nhits",
    target_mode: str = "sum",
    hidden_size: int = 512,
    num_layers: int = 3,
    patch_length: int = 8,
    stride: int = 4,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    holdout_count: int = 12,
) -> dict[str, object]:
    """Train pump.fun model on minute candles."""
    result = train_pumpfun_model(
        model_dir=model_dir,
        horizon_minutes=horizon_minutes,
        context_length=context_length,
        model_type=model_type,
        target_mode=target_mode,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        holdout_count=holdout_count,
    )
    return {
        "model_dir": str(result.model_dir),
        "metrics": result.metrics,
        "holdout_tokens": result.holdout_tokens,
    }


def pumpfun_backtest_command(
    model_dir: str,
    minutes: int = 10,
    test_window: int = 240,
    target_mode: str = "sum",
    max_tokens: int | None = None,
    max_samples: int | None = None,
) -> dict[str, object]:
    """Backtest pump.fun model on holdout tokens."""
    result = backtest_pumpfun_model(
        model_dir=model_dir,
        minutes=minutes,
        test_window=test_window,
        target_mode=target_mode,
        max_tokens=max_tokens,
        max_samples=max_samples,
    )
    return {
        "mae": result.mae,
        "rmse": result.rmse,
        "smape": result.smape,
        "direction_accuracy": result.direction_accuracy,
        "price_accuracy_pct": result.price_accuracy_pct,
        "samples": result.samples,
    }


def pumpfun_predict_command(
    token_id: str,
    model_dir: str,
    minutes: int = 10,
    target_mode: str = "sum",
) -> dict[str, object]:
    """Predict pump.fun token price movement."""
    result = predict_pumpfun(
        token_id=token_id, model_dir=model_dir, minutes=minutes, target_mode=target_mode
    )
    return {
        "token_id": result.token_id,
        "current_price": result.current_price,
        "predicted_price": result.predicted_price,
        "price_change": result.price_change,
        "price_change_pct": result.price_change_pct,
        "direction": result.direction,
        "minutes": result.minutes,
        "confidence": result.confidence,
    }


def pumpfun_classify_train_command(
    model_dir: str,
    horizon_minutes: int = 10,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    label_threshold: float = 0.0,
    holdout_count: int = 12,
    min_token_samples: int = 0,
    use_pos_weight: bool = True,
    normalize_features: bool = True,
) -> dict[str, object]:
    """Train pump.fun direction classifier."""
    result = train_pumpfun_direction_classifier(
        model_dir=model_dir,
        horizon_minutes=horizon_minutes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        label_threshold=label_threshold,
        holdout_count=holdout_count,
        min_token_samples=min_token_samples,
        use_pos_weight=use_pos_weight,
        normalize_features=normalize_features,
    )
    return {
        "model_dir": str(result.model_dir),
        "metrics": result.metrics,
        "holdout_tokens": result.holdout_tokens,
    }


def pumpfun_classify_backtest_command(
    model_dir: str,
    max_tokens: int | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Backtest pump.fun direction classifier."""
    return backtest_pumpfun_direction_classifier(
        model_dir=model_dir, max_tokens=max_tokens, max_samples=max_samples
    )
