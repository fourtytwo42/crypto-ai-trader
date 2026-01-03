"""Data preparation utilities for training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """Train, validation, and test splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_by_ratio(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> DataSplits:
    """Split data by ratios.

    Args:
        df: DataFrame sorted by timestamp.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.

    Returns:
        DataSplits with train, val, test.
    """
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("ratios must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return DataSplits(
        train=df.iloc[:train_end].reset_index(drop=True),
        val=df.iloc[train_end:val_end].reset_index(drop=True),
        test=df.iloc[val_end:].reset_index(drop=True),
    )


def to_neuralforecast_format(
    df: pd.DataFrame,
    target_col: str = "return",
    timestamp_col: str = "timestamp",
    unique_id: str = "BTC",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert to NeuralForecast format.

    Returns:
        DataFrame with columns [unique_id, ds, y].
    """
    if timestamp_col not in df.columns or target_col not in df.columns:
        raise ValueError("missing required columns")
    data = {
        "unique_id": unique_id,
        "ds": df[timestamp_col],
        "y": df[target_col],
    }
    if feature_cols:
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError("missing feature columns")
        for col in feature_cols:
            data[col] = df[col]
    return pd.DataFrame(
        {
            **data
        }
    )


def walk_forward_splits(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    purge: int = 1,
) -> list[DataSplits]:
    """Generate walk-forward splits.

    Args:
        df: DataFrame sorted by timestamp.
        train_size: Number of rows in train split.
        val_size: Number of rows in validation split.
        test_size: Number of rows in test split.
        step_size: Rows to step forward each iteration.
        purge: Rows to drop between train and validation to reduce leakage.
    """
    if min(train_size, val_size, test_size, step_size) <= 0:
        raise ValueError("sizes must be positive")
    if purge < 0:
        raise ValueError("purge must be non-negative")

    splits: list[DataSplits] = []
    start = 0
    while start + train_size + purge + val_size + test_size <= len(df):
        train_end = start + train_size
        val_start = train_end + purge
        val_end = val_start + val_size
        test_end = val_end + test_size
        splits.append(
            DataSplits(
                train=df.iloc[start:train_end].reset_index(drop=True),
                val=df.iloc[val_start:val_end].reset_index(drop=True),
                test=df.iloc[val_end:test_end].reset_index(drop=True),
            )
        )
        start += step_size
    return splits
