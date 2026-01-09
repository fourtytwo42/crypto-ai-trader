"""Model evaluation utilities."""

from __future__ import annotations

import math

import numpy as np


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dict with mae, rmse, and mape.
    """
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)))
    return {"mae": mae, "rmse": rmse, "mape": mape}
