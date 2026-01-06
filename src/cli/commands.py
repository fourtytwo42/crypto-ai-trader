"""CLI command implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog
import numpy as np
import json

from src.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from src.config import get_settings
from src.data.pipeline import DataPipeline, prepare_training_data, prepare_training_data_multi
from src.data.feature_extractor import get_feature_columns
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


def menu_command() -> None:
    """Launch interactive menu."""
    from src.cli.menu import run_menu

    run_menu()
