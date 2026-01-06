"""Training configuration for Bitcoin Trading Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ModelType = Literal["patchtst", "nhits"]


DataFrequency = Literal["1day", "1hour", "4hour", "6hour", "8hour"]


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: ModelType = "patchtst"
    context_length: int = 128
    horizon: int = 1  # Prediction horizon in periods (days for daily, hours for hourly)
    hidden_size: int = 512
    num_layers: int = 6
    patch_length: int = 16
    stride: int = 8
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cuda"
    freq: str = "H"  # NeuralForecast frequency: "D" for daily, "H" for hourly
    max_vram_gb: float | None = 23.0
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    horizon_weight: list[float] | None = None
    loss_type: str = "mae"
    data_frequency: DataFrequency = "1day"  # KuCoin timeframe for data fetching

    def to_dict(self) -> dict[str, float | int | str | list[float]]:
        """Convert config to a JSON-serializable dict."""
        return {
            "model_type": self.model_type,
            "context_length": self.context_length,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "device": self.device,
            "freq": self.freq,
            "max_vram_gb": self.max_vram_gb,
            "quantiles": list(self.quantiles),
            "horizon_weight": list(self.horizon_weight) if self.horizon_weight else None,
            "loss_type": self.loss_type,
            "data_frequency": self.data_frequency,
        }

    def validate(self) -> None:
        """Validate config values."""
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.model_type not in {"patchtst", "nhits"}:
            raise ValueError("model_type must be 'patchtst' or 'nhits'")
        if len(self.quantiles) != 3:
            raise ValueError("quantiles must have three entries")
        if sorted(self.quantiles) != list(self.quantiles):
            raise ValueError("quantiles must be sorted")
        if not self.freq:
            raise ValueError("freq must be non-empty")
        if self.max_vram_gb is not None and self.max_vram_gb <= 0:
            raise ValueError("max_vram_gb must be positive when set")
        valid_frequencies = {"1day", "1hour", "4hour", "6hour", "8hour"}
        if self.data_frequency not in valid_frequencies:
            raise ValueError(f"data_frequency must be one of {valid_frequencies}")
        if self.horizon_weight is not None and len(self.horizon_weight) != self.horizon:
            raise ValueError("horizon_weight length must match horizon")
        if self.loss_type not in {"mae", "huber"}:
            raise ValueError("loss_type must be 'mae' or 'huber'")
