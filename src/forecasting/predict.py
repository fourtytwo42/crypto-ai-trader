"""Forecast prediction helpers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.training.data_preparation import to_neuralforecast_format
from src.training.model_factory import SimpleQuantileModel


def _extract_forecast_values(forecast: pd.DataFrame) -> np.ndarray:
    columns = list(forecast.columns)
    if "y" in columns:
        values = forecast["y"].to_numpy(dtype=float)
    else:
        value_cols = [col for col in columns if col not in {"ds", "unique_id"}]
        if not value_cols:
            raise ValueError("forecast output missing prediction columns")
        values = forecast[value_cols[0]].to_numpy(dtype=float)
    return values


def forecast_next_horizon(
    model: object,
    history_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    horizon: int,
) -> np.ndarray:
    """Forecast next horizon steps for a target column."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if isinstance(model, SimpleQuantileModel):
        preds = model.predict(horizon)["q50"]
        return np.asarray(preds, dtype=float)

    nf_data = to_neuralforecast_format(
        history_df,
        target_col=target_col,
        feature_cols=feature_cols,
    )
    forecast = model.predict(nf_data)
    values = _extract_forecast_values(forecast)
    if values.size >= horizon:
        return values[-horizon:]
    raise ValueError("forecast output shorter than horizon")


def build_future_timestamps(last_timestamp: datetime, horizon: int) -> list[datetime]:
    """Build future hourly timestamps starting after last timestamp."""
    start = pd.to_datetime(last_timestamp, utc=True)
    future = pd.date_range(start=start, periods=horizon + 1, freq="H", tz="UTC")[1:]
    return [ts.to_pydatetime() for ts in future]
