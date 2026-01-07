"""Helpers for recent KuCoin feature preparation."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.feature_extractor import extract_features_hourly
from src.data.kucoin_client import backfill_kucoin_candles
from src.data.normalizer import RollingNormalizer


def fetch_recent_hourly_candles(symbol: str, hours_back: int) -> list:
    """Fetch the most recent hourly candles for a symbol."""
    end_at = int(time.time())
    start_at = end_at - max(hours_back, 1) * 3600
    return list(
        backfill_kucoin_candles(
            symbol=symbol,
            timeframe="1hour",
            start_at=start_at,
            end_at=end_at,
        )
    )


def build_recent_feature_df_from_candles(
    candles: list,
    min_rows: int,
    normalize_cols: list[str],
) -> tuple[pd.DataFrame, RollingNormalizer | None]:
    """Convert raw candles into normalized feature DataFrame."""
    if len(candles) < min_rows:
        return pd.DataFrame(), None
    rows = []
    for candle in candles:
        rows.append(
            {
                "timestamp": datetime.fromtimestamp(candle.timestamp, tz=timezone.utc),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), None
    features_df = extract_features_hourly(df, drop_na=True, daily_equivalent=True)
    if features_df.empty:
        return pd.DataFrame(), None
    hour = pd.to_datetime(features_df["timestamp"], utc=True).dt.hour
    dow = pd.to_datetime(features_df["timestamp"], utc=True).dt.dayofweek
    two_pi = 2 * np.pi
    features_df["hour_sin"] = np.sin(two_pi * hour / 24.0)
    features_df["hour_cos"] = np.cos(two_pi * hour / 24.0)
    features_df["dow_sin"] = np.sin(two_pi * dow / 7.0)
    features_df["dow_cos"] = np.cos(two_pi * dow / 7.0)
    eps = 1e-10
    features_df["log_close"] = np.log(features_df["close"].clip(lower=eps))
    features_df["log_volume"] = np.log(features_df["volume"].clip(lower=eps))
    normalizer = RollingNormalizer(window=30)
    normalized_df = normalizer.fit_transform(features_df, normalize_cols)
    return normalized_df, normalizer
