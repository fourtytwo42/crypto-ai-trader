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


def _disable_model_logging(model: object) -> None:
    """Disable logging for NeuralForecast models during prediction to avoid FileExistsError."""
    try:
        from neuralforecast import NeuralForecast
        if not isinstance(model, NeuralForecast):
            return
        
        # Access the underlying models and disable their loggers
        models = getattr(model, "models", [])
        for submodel in models:
            # Get trainer kwargs and set logger to False
            trainer_kwargs = getattr(submodel, "trainer_kwargs", None)
            if isinstance(trainer_kwargs, dict):
                trainer_kwargs["logger"] = False
            # Also try to set it directly on the model if it has a trainer attribute
            trainer = getattr(submodel, "trainer", None)
            if trainer is not None:
                # Disable logger by setting it to False
                if hasattr(trainer, "logger"):
                    trainer.logger = False
    except Exception:
        # If we can't disable logging, continue anyway
        pass


def forecast_next_horizon(model: object, history_df: pd.DataFrame, target_col: str, feature_cols: list[str], horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    
    # Disable logging to prevent FileExistsError during prediction
    _disable_model_logging(model)
    
    nf_data = to_neuralforecast_format(history_df, target_col=target_col, feature_cols=feature_cols)
    
    # Wrap predict call to handle FileExistsError from tensorboardX logger
    # This happens when the logger tries to create a directory that already exists
    # It's safe to retry - the directory existing is not a problem
    max_retries = 3
    for attempt in range(max_retries):
        try:
            forecast = model.predict(nf_data)
            break
        except FileExistsError as e:
            # If the error is about lightning_logs/version_* directory, it's a race condition
            # The directory was created by another thread/process
            # This is safe to ignore - retry the prediction
            if "lightning_logs" in str(e) or "version_" in str(e):
                if attempt < max_retries - 1:
                    # Brief sleep to let the directory creation complete
                    import time
                    time.sleep(0.05 * (attempt + 1))
                    continue
                else:
                    # Last attempt failed - the directory should exist now, try once more
                    forecast = model.predict(nf_data)
            else:
                # Not a lightning_logs error, re-raise
                raise
        except Exception:
            # Other errors should be raised normally
            raise
    
    values = _extract_forecast_values(forecast)
    if values.size >= horizon:
        return values[-horizon:]
    raise ValueError("forecast output shorter than horizon")
