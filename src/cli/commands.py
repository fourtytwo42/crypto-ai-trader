"""CLI command implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from src.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from src.config import get_settings
from src.data.pipeline import DataPipeline, prepare_training_data
from src.database.connection import get_db_manager
from src.database.operations import get_latest_features
from src.prediction.model_loader import load_model_artifacts
from src.prediction.predictor import generate_predictions
from src.prediction.signal_generator import SignalConfig, generate_signal
from src.training.config import TrainingConfig
from src.training.trainer import train_model

logger = structlog.get_logger(__name__)


def load_data_command(file_path: str, replace_existing: bool = False) -> dict[str, int]:
    """Load CSV data into the database."""
    db = get_db_manager()
    with db.session() as session:
        pipeline = DataPipeline(session)
        return pipeline.run_full_pipeline(file_path, replace_existing=replace_existing)


def train_command(model_dir: str | None = None) -> dict[str, float]:
    """Train a model using features in the database."""
    settings = get_settings()
    model_dir = model_dir or settings.model_dir

    db = get_db_manager()
    with db.session() as session:
        features_df, _ = prepare_training_data(session)

    if features_df.empty:
        raise ValueError("no features available for training")

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
            features = get_latest_features(session, limit=1)
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
    horizon: int = 1,
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

    # Build backtest config with enhanced options
    backtest_config = BacktestConfig(
        threshold=settings.signal_threshold,
        volatility_adaptive=volatility_adaptive,
        volatility_window=settings.signal_volatility_window,
        max_uncertainty_spread=max_uncertainty_spread,
        uncertainty_enabled=uncertainty_enabled,
        min_holding_periods=min_holding_periods,
    )

    result = walk_forward_backtest(
        data,
        config,
        train_size=20,
        val_size=5,
        test_size=5,
        step_size=5,
        model_dir=model_dir,
        backtest_config=backtest_config,
    )
    return result.aggregated


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
