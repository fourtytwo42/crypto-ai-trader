from __future__ import annotations

import pandas as pd
import numpy as np

from features import extract_features_minute
from normalizer import RollingNormalizer


def _timestamp_to_utc(ts: int | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    unit = "ms" if ts > 1_000_000_000_000 else "s"
    return pd.to_datetime(ts, unit=unit, utc=True)


def build_recent_feature_df(candles_df: pd.DataFrame, token_meta: dict[str, object]) -> pd.DataFrame:
    if candles_df.empty:
        return pd.DataFrame()
    candles_df = candles_df.sort_values("timestamp").reset_index(drop=True)
    features_df = extract_features_minute(candles_df)
    if features_df.empty:
        return pd.DataFrame()

    eps = 1e-10
    features_df["log_close"] = np.log(features_df["close"].clip(lower=eps))
    features_df["log_volume"] = np.log(features_df["volume_usd"].clip(lower=eps))
    if "trades" in features_df.columns:
        features_df["log_trades"] = np.log(features_df["trades"].clip(lower=1.0))

    created_dt = _timestamp_to_utc(int(token_meta.get("created_timestamp")))
    koth_ts = token_meta.get("king_of_the_hill_timestamp")
    koth_dt = _timestamp_to_utc(int(koth_ts)) if koth_ts else None
    completed = bool(token_meta.get("completed", False))

    features_df["minutes_since_launch"] = (
        (features_df["timestamp"] - created_dt).dt.total_seconds() / 60 if created_dt is not None else 0.0
    )
    if koth_dt is not None:
        features_df["minutes_since_koth"] = (features_df["timestamp"] - koth_dt).dt.total_seconds() / 60
        features_df["has_koth"] = 1.0
        features_df["koth_reached"] = (features_df["timestamp"] >= koth_dt).astype(float)
    else:
        features_df["minutes_since_koth"] = 0.0
        features_df["has_koth"] = 0.0
        features_df["koth_reached"] = 0.0
    features_df["is_completed"] = 1.0 if completed else 0.0

    normalize_cols = [
        "range",
        "body",
        "dlog_volume",
        "ret_mean_5",
        "ret_std_5",
        "ret_mean_15",
        "ret_std_15",
        "ret_mean_60",
        "ret_std_60",
        "minutes_since_launch",
        "minutes_since_koth",
        "log_trades",
        "log_close",
        "log_volume",
    ]
    normalizer = RollingNormalizer(window=60)
    features_df = normalizer.fit_transform(features_df, normalize_cols)

    return features_df
