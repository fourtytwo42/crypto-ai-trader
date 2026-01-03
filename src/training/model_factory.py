"""Model factory for training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.training.config import TrainingConfig


class ModelFactoryError(RuntimeError):
    """Raised when model creation fails."""


@dataclass
class SimpleQuantileModel:
    """Simple quantile model used for tests and fallback."""

    quantiles: list[float]
    mean_: float | None = None
    std_: float | None = None

    def fit(self, y: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("training data is empty")
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y))

    def predict(self, steps: int) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("model must be fit before prediction")
        z_values = {
            0.1: -1.2815515655446004,
            0.5: 0.0,
            0.9: 1.2815515655446004,
        }
        preds: dict[str, np.ndarray] = {}
        for q in self.quantiles:
            z = z_values.get(float(q), 0.0)
            preds[f"q{int(q * 100)}"] = np.full(steps, self.mean_ + z * self.std_)
        return preds


def create_model(
    config: TrainingConfig,
    force_simple: bool = False,
    feature_cols: list[str] | None = None,
) -> Any:
    """Create a model instance based on config.

    Args:
        config: TrainingConfig.
        force_simple: If True, return SimpleQuantileModel.
        feature_cols: Optional list of historical exogenous feature columns.

    Returns:
        Model instance with fit/predict.
    """
    config.validate()
    if force_simple:
        return SimpleQuantileModel(quantiles=config.quantiles)

    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS, PatchTST
        from neuralforecast.losses.pytorch import MAE, QuantileLoss
        import torch
    except Exception:
        return SimpleQuantileModel(quantiles=config.quantiles)

    use_quantiles = config.horizon == 1
    quantile_loss = QuantileLoss(q=torch.tensor(config.quantiles)) if use_quantiles else None
    base_loss = MAE() if not use_quantiles else None

    if config.model_type == "patchtst":
        model = PatchTST(
            h=config.horizon,
            input_size=config.context_length,
            hidden_size=config.hidden_size,
            encoder_layers=config.num_layers,
            patch_len=config.patch_length,
            stride=config.stride,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            max_steps=config.epochs,
            loss=quantile_loss or base_loss,
            valid_loss=quantile_loss or base_loss,
        )
    elif config.model_type == "nhits":
        model = NHITS(
            h=config.horizon,
            input_size=config.context_length,
            hist_exog_list=feature_cols,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            max_steps=config.epochs,
            loss=quantile_loss or base_loss,
            valid_loss=quantile_loss or base_loss,
        )
    else:
        raise ModelFactoryError(f"unsupported model_type: {config.model_type}")

    return NeuralForecast(models=[model], freq=config.freq)
