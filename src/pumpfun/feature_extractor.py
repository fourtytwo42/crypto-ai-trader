"""Feature extraction for pump.fun minute candles."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return",
    "range",
    "body",
    "dlog_volume",
    "ret_mean_15",
    "ret_std_15",
    "ret_mean_60",
    "ret_std_60",
]


class PumpFeatureExtractionError(ValueError):
    """Raised when pump.fun feature extraction fails."""


def extract_features_minute(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    if df.empty:
        raise PumpFeatureExtractionError("DataFrame is empty")

    missing = [col for col in ["open", "high", "low", "close", "volume_usd"] if col not in df.columns]
    if missing:
        raise PumpFeatureExtractionError(f"Missing required columns: {missing}")

    result = df.copy()
    result["return"] = np.log(result["close"] / result["close"].shift(1))
    result["range"] = np.log(result["high"] / result["low"])
    result["body"] = np.log(result["close"] / result["open"])

    volume = result["volume_usd"].replace(0, 1e-10)
    result["dlog_volume"] = np.log(volume) - np.log(volume.shift(1))

    result["ret_mean_15"] = result["return"].rolling(window=15).mean()
    result["ret_std_15"] = result["return"].rolling(window=15).std()
    result["ret_mean_60"] = result["return"].rolling(window=60).mean()
    result["ret_std_60"] = result["return"].rolling(window=60).std()

    if drop_na:
        result = result.dropna().reset_index(drop=True)

    return result
