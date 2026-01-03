"""Training orchestration."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from src.training.config import TrainingConfig
from src.training.evaluator import calculate_metrics
from src.training.model_factory import SimpleQuantileModel, create_model

logger = structlog.get_logger(__name__)


@dataclass
class TrainingResult:
    """Result from model training."""

    model: object
    metrics: dict[str, float]
    model_path: Path
    metadata_path: Path
    scaler_path: Path | None


def _is_existing_better(existing: dict[str, float], new: dict[str, float]) -> bool:
    """Return True if existing metrics are better than new metrics."""
    for key in ("mae", "rmse", "mape"):
        if key in existing and key in new:
            return existing[key] <= new[key]
    return False


def _select_model_dir(model_dir: str | Path, metrics: dict[str, float], keep_best: bool) -> Path:
    model_dir = Path(model_dir)
    if not keep_best:
        return model_dir

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return model_dir

    try:
        existing_metrics = json.loads(metadata_path.read_text()).get("metrics", {})
    except Exception:
        return model_dir

    if _is_existing_better(existing_metrics, metrics):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return model_dir / "runs" / timestamp

    return model_dir


def _prepare_target(train_df: pd.DataFrame, target_col: str) -> np.ndarray:
    if target_col not in train_df.columns:
        raise ValueError(f"missing target column: {target_col}")
    return train_df[target_col].to_numpy(dtype=float)


def train_model(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: str | Path,
    target_col: str = "return",
    force_simple: bool = False,
    scaler: object | None = None,
    feature_cols: list[str] | None = None,
    save_artifacts: bool = True,
    keep_best: bool = False,
) -> TrainingResult:
    """Train a model and persist artifacts.

    Args:
        config: TrainingConfig.
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        model_dir: Directory to save model artifacts.
        target_col: Target column name.
        force_simple: Use SimpleQuantileModel even if NeuralForecast exists.
    """
    model = create_model(config, force_simple=force_simple, feature_cols=feature_cols)
    y_train = _prepare_target(train_df, target_col)

    if isinstance(model, SimpleQuantileModel):
        model.fit(y_train)
        y_val = val_df[target_col].to_numpy(dtype=float)
        preds = model.predict(len(y_val))["q50"]
        metrics = calculate_metrics(y_val, preds)
    else:
        # NeuralForecast expects a data frame with columns [unique_id, ds, y]
        from src.training.data_preparation import to_neuralforecast_format

        train_nf = to_neuralforecast_format(
            train_df, target_col=target_col, feature_cols=feature_cols
        )
        val_nf = to_neuralforecast_format(
            val_df, target_col=target_col, feature_cols=feature_cols
        )
        model.fit(train_nf)
        forecasts = model.predict(val_nf)
        preds = forecasts["y"] if "y" in forecasts.columns else forecasts.iloc[:, 1]
        y_true = val_nf["y"].to_numpy(dtype=float)
        y_pred = preds.to_numpy(dtype=float)
        if y_pred.ndim > 1:
            y_pred = y_pred[:, 0]
        if y_pred.shape[0] != y_true.shape[0]:
            min_len = min(y_pred.shape[0], y_true.shape[0])
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        metrics = calculate_metrics(y_true, y_pred)

    model_path = Path(model_dir) / "model.pt"
    metadata_path = Path(model_dir) / "metadata.json"
    if save_artifacts:
        model_path, metadata_path = save_model_artifacts(
            model,
            config,
            metrics,
            model_dir=model_dir,
            scaler=scaler,
            keep_best=keep_best,
        )

    return TrainingResult(
        model=model,
        metrics=metrics,
        model_path=model_path,
        metadata_path=metadata_path,
        scaler_path=None,
    )


def save_model_artifacts(
    model: object,
    config: TrainingConfig,
    metrics: dict[str, float],
    model_dir: str | Path,
    scaler: object | None = None,
    keep_best: bool = False,
) -> tuple[Path, Path]:
    """Save model and metadata to disk."""
    model_dir = _select_model_dir(model_dir, metrics, keep_best)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    scaler_path = model_dir / "scaler.pkl"

    saved = False
    try:
        import torch

        torch.save(model, model_path)
        saved = True
    except Exception:
        saved = False

    if not saved:
        with model_path.open("wb") as handle:
            pickle.dump(model, handle)

    if scaler is not None:
        with scaler_path.open("wb") as handle:
            pickle.dump(scaler, handle)

    metadata = {
        "config": config.to_dict(),
        "metrics": metrics,
        "scaler_path": str(scaler_path) if scaler_path.exists() else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model artifacts", model_path=str(model_path))

    return model_path, metadata_path
