"""CLI command implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog

from src.backtest.walk_forward import walk_forward_backtest
from src.config import get_settings
from src.data.pipeline import DataPipeline, prepare_training_data
from src.database.connection import get_db_manager
from src.database.operations import get_latest_features
from src.prediction.model_loader import load_model_artifacts
from src.prediction.predictor import generate_predictions
from src.prediction.signal_generator import generate_signal
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
    )
    return result.metrics


def predict_command(model_dir: str, data: pd.DataFrame | None = None) -> dict[str, object]:
    """Generate a prediction using latest DB features or provided data."""
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

    predictions = generate_predictions(bundle.model, data)
    latest = predictions[-1]
    signal = generate_signal({"q10": latest["q10"], "q50": latest["q50"], "q90": latest["q90"]})
    return {"prediction": latest, "signal": signal}


def backtest_command(model_dir: str, data: pd.DataFrame | None = None) -> dict[str, float]:
    """Run a walk-forward backtest."""
    config = TrainingConfig(model_type="patchtst")
    if data is None:
        db = get_db_manager()
        with db.session() as session:
            features_df, _ = prepare_training_data(session, normalize=False)
        if features_df.empty:
            raise ValueError("no features available for backtest")
        data = features_df

    result = walk_forward_backtest(
        data,
        config,
        train_size=20,
        val_size=5,
        test_size=5,
        step_size=5,
        model_dir=model_dir,
    )
    return result.aggregated


def menu_command() -> None:
    """Launch interactive menu."""
    from src.cli.menu import run_menu

    run_menu()
