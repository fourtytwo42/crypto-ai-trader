"""Target engineering helpers for training."""

from __future__ import annotations

import pandas as pd


def add_return_24h_target(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    """Add 24h log-return target column."""
    df = df.copy()
    if symbol_col in df.columns:
        df = df.sort_values([symbol_col, "timestamp"]).reset_index(drop=True)
        df["return_24h"] = df.groupby(symbol_col)["log_close"].shift(-24) - df["log_close"]
    else:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["return_24h"] = df["log_close"].shift(-24) - df["log_close"]
    return df
