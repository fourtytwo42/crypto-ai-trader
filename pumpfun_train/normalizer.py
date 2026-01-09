"""Normalization utilities for Bitcoin Trading Model.

Provides rolling z-score normalization to prevent data leakage
while handling non-stationarity in financial time series.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Default rolling window for normalization
DEFAULT_WINDOW = 30


class NormalizationError(Exception):
    """Error during normalization."""

    pass


class RollingNormalizer:
    """Rolling z-score normalizer.

    Uses a rolling window to compute mean and std, ensuring only
    past data is used for normalization (no lookahead bias).
    """

    def __init__(self, window: int = DEFAULT_WINDOW, min_periods: int | None = None):
        """Initialize normalizer.

        Args:
            window: Rolling window size in periods.
            min_periods: Minimum periods for valid calculation.
                        Defaults to window size.
        """
        self.window = window
        self.min_periods = min_periods or window
        self._fitted = False
        self._feature_stats: dict[str, dict[str, float]] = {}

    def fit(self, df: pd.DataFrame, columns: list[str] | None = None) -> "RollingNormalizer":
        """Fit normalizer on training data.

        Stores the last window of data for each feature to enable
        consistent normalization of new data.

        Args:
            df: Training DataFrame.
            columns: Columns to normalize. Defaults to all numeric columns.

        Returns:
            Self for method chaining.
        """
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            # Store rolling stats from end of training data
            rolling_mean = df[col].rolling(window=self.window, min_periods=self.min_periods).mean()
            rolling_std = df[col].rolling(window=self.window, min_periods=self.min_periods).std()

            # Store last valid stats for inference
            last_mean = rolling_mean.iloc[-1] if not rolling_mean.empty else 0.0
            last_std = rolling_std.iloc[-1] if not rolling_std.empty else 1.0

            self._feature_stats[col] = {
                "mean": float(last_mean),
                "std": float(last_std) if last_std > 0 else 1.0,
            }

        self._fitted = True
        # Don't log during training - too noisy
        # logger.debug("Normalizer fitted", features=list(self._feature_stats.keys()))
        return self

    def transform(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Transform data using rolling z-score.

        For training data, uses rolling statistics.
        For inference, can use stored statistics from fit().

        Args:
            df: DataFrame to transform.
            columns: Columns to normalize. Defaults to all numeric columns.

        Returns:
            Normalized DataFrame.
        """
        columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()

        result = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            # Use rolling normalization
            rolling_mean = df[col].rolling(
                window=self.window, min_periods=self.min_periods
            ).mean()
            rolling_std = df[col].rolling(
                window=self.window, min_periods=self.min_periods
            ).std()

            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1.0)

            result[col] = (df[col] - rolling_mean) / rolling_std

        return result

    def transform_single(
        self, values: dict[str, float] | pd.Series
    ) -> dict[str, float]:
        """Transform a single data point using stored statistics.

        Useful for inference when normalizing a single new observation.

        Args:
            values: Dictionary or Series with feature values.

        Returns:
            Normalized values as dictionary.

        Raises:
            NormalizationError: If normalizer not fitted.
        """
        if not self._fitted:
            raise NormalizationError("Normalizer not fitted. Call fit() first.")

        if isinstance(values, pd.Series):
            values = values.to_dict()

        result = {}
        for col, value in values.items():
            if col in self._feature_stats:
                stats = self._feature_stats[col]
                result[col] = (value - stats["mean"]) / stats["std"]
            else:
                result[col] = value

        return result

    def fit_transform(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.
            columns: Columns to normalize.

        Returns:
            Normalized DataFrame.
        """
        self.fit(df, columns)
        return self.transform(df, columns)

    def save(self, path: str | Path) -> None:
        """Save normalizer state to file.

        Args:
            path: File path for pickle file.
        """
        path = Path(path)
        state = {
            "window": self.window,
            "min_periods": self.min_periods,
            "feature_stats": self._feature_stats,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Normalizer saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "RollingNormalizer":
        """Load normalizer from file.

        Args:
            path: Path to pickle file.

        Returns:
            Loaded normalizer.
        """
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)

        normalizer = cls(window=state["window"], min_periods=state["min_periods"])
        normalizer._feature_stats = state["feature_stats"]
        normalizer._fitted = state["fitted"]

        logger.info("Normalizer loaded", path=str(path))
        return normalizer

    @property
    def is_fitted(self) -> bool:
        """Check if normalizer is fitted."""
        return self._fitted

    @property
    def feature_stats(self) -> dict[str, dict[str, float]]:
        """Get stored feature statistics."""
        return self._feature_stats.copy()


def normalize_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Convenience function for rolling z-score normalization.

    Args:
        df: DataFrame to normalize.
        columns: Feature columns to normalize.
        window: Rolling window size.

    Returns:
        Normalized DataFrame.
    """
    normalizer = RollingNormalizer(window=window)
    return normalizer.transform(df, columns)
