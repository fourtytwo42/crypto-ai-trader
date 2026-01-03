"""Signal generation logic."""

from __future__ import annotations

from typing import Literal


Signal = Literal["long", "short", "flat"]


def generate_signal(prediction: dict[str, float], threshold: float = 0.005) -> Signal:
    """Generate a trading signal from quantile prediction."""
    q10 = float(prediction["q10"])
    q50 = float(prediction["q50"])
    q90 = float(prediction["q90"])

    if q50 > threshold and q10 > -threshold:
        return "long"
    if q50 < -threshold and q90 < threshold:
        return "short"
    return "flat"


def apply_signals(
    predictions: list[dict[str, float]], threshold: float = 0.005
) -> list[Signal]:
    """Apply signal generation to a list of predictions."""
    return [generate_signal(pred, threshold=threshold) for pred in predictions]
