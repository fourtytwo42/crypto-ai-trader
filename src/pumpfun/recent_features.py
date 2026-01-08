"""Recent feature preparation for pump.fun inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.normalizer import RollingNormalizer
from src.pumpfun.feature_extractor import extract_features_minute
from src.pumpfun.data import PUMPFUN_NORMALIZED_COLUMNS, _timestamp_to_utc


def _append_token_meta(features_df: pd.DataFrame, token_meta: dict[str, object]) -> pd.DataFrame:
    features_df = features_df.copy()
    created_ts = token_meta.get("created_timestamp")
    koth_ts = token_meta.get("king_of_the_hill_timestamp")
    completed = bool(token_meta.get("completed", False))
    created_dt = _timestamp_to_utc(int(created_ts)) if created_ts is not None else None
    koth_dt = _timestamp_to_utc(int(koth_ts)) if koth_ts is not None else None
    if created_dt is not None:
        features_df["minutes_since_launch"] = (features_df["timestamp"] - created_dt).dt.total_seconds() / 60
    else:
        features_df["minutes_since_launch"] = 0.0
    if koth_dt is not None:
        features_df["minutes_since_koth"] = (features_df["timestamp"] - koth_dt).dt.total_seconds() / 60
        features_df["has_koth"] = 1.0
        features_df["koth_reached"] = (features_df["timestamp"] >= koth_dt).astype(float)
    else:
        features_df["minutes_since_koth"] = 0.0
        features_df["has_koth"] = 0.0
        features_df["koth_reached"] = 0.0
    features_df["is_completed"] = 1.0 if completed else 0.0
    return features_df


def build_recent_feature_df(
    candles_df: pd.DataFrame,
    token_meta: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, RollingNormalizer]:
    if candles_df.empty:
        return pd.DataFrame(), RollingNormalizer()
    candles_df = candles_df.sort_values("timestamp").reset_index(drop=True)
    features_df = extract_features_minute(candles_df, drop_na=True)
    if features_df.empty:
        return pd.DataFrame(), RollingNormalizer()

    eps = 1e-10
    features_df["log_close"] = np.log(features_df["close"].clip(lower=eps))
    features_df["log_volume"] = np.log(features_df["volume_usd"].clip(lower=eps))
    if "trades" in features_df.columns:
        features_df["log_trades"] = np.log(features_df["trades"].clip(lower=1.0))
    if "return" in features_df.columns:
        features_df["ret_mean_5"] = features_df["return"].rolling(window=5).mean()
        features_df["ret_std_5"] = features_df["return"].rolling(window=5).std()
        features_df = features_df.dropna().reset_index(drop=True)

    if token_meta is not None:
        features_df = _append_token_meta(features_df, token_meta)
    normalizer = RollingNormalizer(window=60)
    normalize_cols = [*PUMPFUN_NORMALIZED_COLUMNS, "log_close", "log_volume"]
    features_df = normalizer.fit_transform(features_df, normalize_cols)
    return features_df, normalizer
