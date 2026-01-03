"""Training configuration for Bitcoin Trading Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ModelType = Literal["patchtst", "nhits"]


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_type: ModelType = "patchtst"
    context_length: int = 128
    horizon: int = 1
    hidden_size: int = 512
    num_layers: int = 6
    patch_length: int = 16
    stride: int = 8
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cuda"
    freq: str = "H"
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

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
            "quantiles": list(self.quantiles),
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
