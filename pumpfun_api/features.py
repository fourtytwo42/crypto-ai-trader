from __future__ import annotations

import numpy as np
import pandas as pd


def extract_features_minute(df: pd.DataFrame, allow_sparse: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    result["return"] = np.log(result["close"] / result["close"].shift(1))
    result["range"] = np.log(result["high"] / result["low"])
    result["body"] = np.log(result["close"] / result["open"])

    volume = result["volume_usd"].replace(0, 1e-10)
    result["dlog_volume"] = np.log(volume) - np.log(volume.shift(1))

    min_periods = 1 if allow_sparse else None
    result["ret_mean_5"] = result["return"].rolling(window=5, min_periods=min_periods).mean()
    result["ret_std_5"] = result["return"].rolling(window=5, min_periods=min_periods).std()
    result["ret_mean_15"] = result["return"].rolling(window=15, min_periods=min_periods).mean()
    result["ret_std_15"] = result["return"].rolling(window=15, min_periods=min_periods).std()
    result["ret_mean_60"] = result["return"].rolling(window=60, min_periods=min_periods).mean()
    result["ret_std_60"] = result["return"].rolling(window=60, min_periods=min_periods).std()

    if allow_sparse:
        return result.fillna(0.0).reset_index(drop=True)
    return result.dropna().reset_index(drop=True)
