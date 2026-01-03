"""Prediction generation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.training.model_factory import SimpleQuantileModel


def generate_predictions(
    model: object,
    data: pd.DataFrame,
    timestamp_col: str = "timestamp",
    feature_cols: list[str] | None = None,
    target_col: str = "return",
) -> list[dict[str, float | datetime]]:
    """Generate quantile predictions for each row in data."""
    if timestamp_col not in data.columns:
        raise ValueError("timestamp column missing")
    steps = len(data)
    if steps == 0:
        return []

    if isinstance(model, SimpleQuantileModel):
        preds = model.predict(steps)
        result: list[dict[str, float | datetime]] = []
        for idx, ts in enumerate(data[timestamp_col]):
            result.append(
                {
                    "timestamp": ts,
                    "q10": float(preds["q10"][idx]),
                    "q50": float(preds["q50"][idx]),
                    "q90": float(preds["q90"][idx]),
                }
            )
        return result

    from src.training.data_preparation import to_neuralforecast_format

    if feature_cols is None:
        from src.data.feature_extractor import get_feature_columns

        feature_cols = [col for col in get_feature_columns() if col in data.columns]
    nf_data = to_neuralforecast_format(
        data, target_col=target_col, feature_cols=feature_cols
    )
    forecast = model.predict(nf_data)
    if len(forecast) == 0:
        return []
    if len(forecast) < len(data):
        data = data.tail(len(forecast)).reset_index(drop=True)
    columns = list(forecast.columns)
    value_col = None
    if {"q10", "q50", "q90"}.issubset(columns):
        value_col = None
    else:
        for col in columns:
            if col != "ds":
                value_col = col
                break
        if value_col is None:
            raise ValueError("forecast output missing prediction columns")
    predictions: list[dict[str, float | datetime]] = []
    for idx, ts in enumerate(data[timestamp_col]):
        if value_col is None:
            q10 = float(forecast.iloc[idx]["q10"])
            q50 = float(forecast.iloc[idx]["q50"])
            q90 = float(forecast.iloc[idx]["q90"])
        else:
            q50 = float(forecast.iloc[idx][value_col])
            q10 = q50
            q90 = q50
        predictions.append(
            {
                "timestamp": ts,
                "q10": q10,
                "q50": q50,
                "q90": q90,
            }
        )
    return predictions


def extract_quantiles(predictions: list[dict[str, float | datetime]]) -> np.ndarray:
    """Extract q50 predictions as numpy array."""
    return np.array([float(item["q50"]) for item in predictions], dtype=float)
