from __future__ import annotations

import numpy as np
import pandas as pd


def to_neuralforecast_format(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
    unique_id: str = "pumpfun",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    data = {
        "unique_id": unique_id,
        "ds": df[timestamp_col],
        "y": df[target_col],
    }
    if feature_cols:
        for col in feature_cols:
            data[col] = df[col]
    return pd.DataFrame(data)


def _extract_forecast_values(forecast: pd.DataFrame) -> np.ndarray:
    columns = list(forecast.columns)
    if "y" in columns:
        return forecast["y"].to_numpy(dtype=float)
    value_cols = [col for col in columns if col not in {"ds", "unique_id"}]
    if not value_cols:
        raise ValueError("forecast output missing prediction columns")
    return forecast[value_cols[0]].to_numpy(dtype=float)


def forecast_next_horizon(model: object, history_df: pd.DataFrame, target_col: str, feature_cols: list[str], horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    nf_data = to_neuralforecast_format(history_df, target_col=target_col, feature_cols=feature_cols)
    forecast = model.predict(nf_data)
    values = _extract_forecast_values(forecast)
    if values.size >= horizon:
        return values[-horizon:]
    raise ValueError("forecast output shorter than horizon")
