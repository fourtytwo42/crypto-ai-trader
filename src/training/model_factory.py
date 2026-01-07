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
        from neuralforecast.losses.pytorch import MAE, QuantileLoss, HuberLoss
        import torch
        from src.training.progress import EpochProgressCallback
    except Exception:
        return SimpleQuantileModel(quantiles=config.quantiles)

    use_quantiles = config.horizon == 1
    loss_device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    if loss_device == "cuda" and config.max_vram_gb is not None:
        try:
            device_index = torch.cuda.current_device()
            total_mem = torch.cuda.get_device_properties(device_index).total_memory
            max_bytes = int(config.max_vram_gb * 1024 * 1024 * 1024)
            if total_mem > 0:
                fraction = min(1.0, max_bytes / total_mem)
                if fraction > 0:
                    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        except Exception:
            pass
    quantile_loss = (
        QuantileLoss(q=torch.tensor(config.quantiles, device=loss_device))
        if use_quantiles
        else None
    )
    horizon_weight = None
    if config.horizon_weight is not None:
        horizon_weight = np.asarray(config.horizon_weight, dtype=float)
    base_loss = None
    if not use_quantiles:
        if config.loss_type == "mae":
            base_loss = MAE(horizon_weight=horizon_weight)
        elif config.loss_type == "huber":
            base_loss = HuberLoss(delta=1.0, horizon_weight=horizon_weight)
        else:
            raise ModelFactoryError(f"unsupported loss_type: {config.loss_type}")
    use_gpu = loss_device == "cuda"
    trainer_kwargs = {
        "enable_progress_bar": False,
        "callbacks": [EpochProgressCallback(config.epochs)],
        "accelerator": "gpu" if use_gpu else "cpu",
        "devices": 1,
    }
    if use_gpu:
        # Mixed precision reduces activation memory footprint.
        trainer_kwargs["precision"] = "16-mixed"

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
            **trainer_kwargs,
            loss=quantile_loss or base_loss,
            valid_loss=quantile_loss or base_loss,
        )
    elif config.model_type == "nhits":
        nhits_kwargs: dict[str, Any] = {}
        if config.nhits_stack_types is not None:
            nhits_kwargs["stack_types"] = config.nhits_stack_types
        if config.nhits_n_blocks is not None:
            nhits_kwargs["n_blocks"] = config.nhits_n_blocks
        if config.nhits_mlp_units is not None:
            nhits_kwargs["mlp_units"] = config.nhits_mlp_units
        if config.nhits_n_pool_kernel_size is not None:
            nhits_kwargs["n_pool_kernel_size"] = config.nhits_n_pool_kernel_size
        if config.nhits_n_freq_downsample is not None:
            nhits_kwargs["n_freq_downsample"] = config.nhits_n_freq_downsample
        model = NHITS(
            h=config.horizon,
            input_size=config.context_length,
            hist_exog_list=feature_cols,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            max_steps=config.epochs,
            **trainer_kwargs,
            loss=quantile_loss or base_loss,
            valid_loss=quantile_loss or base_loss,
            **nhits_kwargs,
        )
    else:
        raise ModelFactoryError(f"unsupported model_type: {config.model_type}")

    return NeuralForecast(models=[model], freq=config.freq)
