"""Shim for SimpleQuantileModel to satisfy torch load."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimpleQuantileModel:
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
