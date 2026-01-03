"""Feature extraction for Bitcoin Trading Model.

Extracts 8 scale-free features from OHLCV candle data:
1. return: log(C_t / C_{t-1})
2. range: log(H_t / L_t)
3. body: log(C_t / O_t)
4. dlog_volume: log(V_t) - log(V_{t-1})
5. ret_mean_7: 7-day rolling mean of returns
6. ret_std_7: 7-day rolling std of returns
7. ret_mean_30: 30-day rolling mean of returns
8. ret_std_30: 30-day rolling std of returns
"""

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Feature column names
FEATURE_COLUMNS = [
    "return",
    "range",
    "body",
    "dlog_volume",
    "ret_mean_7",
    "ret_std_7",
    "ret_mean_30",
    "ret_std_30",
]

# Required OHLCV columns
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

# Rolling window sizes (daily)
ROLLING_WINDOW_7 = 7
ROLLING_WINDOW_30 = 30

# Rolling window sizes (hourly - daily equivalent)
HOURLY_ROLLING_WINDOW_7 = 7 * 24  # 168 hours = 7 days
HOURLY_ROLLING_WINDOW_30 = 30 * 24  # 720 hours = 30 days


class FeatureExtractionError(Exception):
    """Error during feature extraction."""

    pass


def extract_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Extract scale-free features from candle data.

    Extracts exactly 8 features that are scale-free and suitable
    for time-series forecasting models.

    Args:
        df: DataFrame with OHLCV columns.
        drop_na: If True, remove rows with NaN values (from rolling windows).

    Returns:
        DataFrame with original columns plus 8 feature columns.

    Raises:
        FeatureExtractionError: If required columns are missing or data is empty.
    """
    if df.empty:
        raise FeatureExtractionError("DataFrame is empty")

    # Validate required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise FeatureExtractionError(f"Missing required columns: {missing}")

    logger.info("Extracting features", num_rows=len(df))

    result = df.copy()

    # 1. Log return: log(C_t / C_{t-1})
    result["return"] = np.log(result["close"] / result["close"].shift(1))

    # 2. Range: log(H_t / L_t)
    result["range"] = np.log(result["high"] / result["low"])

    # 3. Body: log(C_t / O_t)
    result["body"] = np.log(result["close"] / result["open"])

    # 4. Volume change: log(V_t) - log(V_{t-1})
    # Handle zero volume by replacing with small value
    volume = result["volume"].replace(0, 1e-10)
    result["dlog_volume"] = np.log(volume) - np.log(volume.shift(1))

    # 5-6. 7-day rolling stats
    result["ret_mean_7"] = result["return"].rolling(window=ROLLING_WINDOW_7).mean()
    result["ret_std_7"] = result["return"].rolling(window=ROLLING_WINDOW_7).std()

    # 7-8. 30-day rolling stats
    result["ret_mean_30"] = result["return"].rolling(window=ROLLING_WINDOW_30).mean()
    result["ret_std_30"] = result["return"].rolling(window=ROLLING_WINDOW_30).std()

    if drop_na:
        # Remove rows with NaN (from rolling windows and shifts)
        initial_count = len(result)
        result = result.dropna().reset_index(drop=True)
        dropped = initial_count - len(result)
        logger.info(
            "Features extracted",
            num_rows=len(result),
            num_features=len(FEATURE_COLUMNS),
            dropped_rows=dropped,
        )
    else:
        logger.info(
            "Features extracted (with NaN)",
            num_rows=len(result),
            num_features=len(FEATURE_COLUMNS),
        )

    return result


def get_feature_columns() -> list[str]:
    """Get list of feature column names.

    Returns:
        List of 8 feature column names.
    """
    return FEATURE_COLUMNS.copy()


def validate_features(df: pd.DataFrame) -> bool:
    """Validate DataFrame has all required feature columns.

    Args:
        df: DataFrame to validate.

    Returns:
        True if all feature columns present.

    Raises:
        FeatureExtractionError: If feature columns are missing.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise FeatureExtractionError(f"Missing feature columns: {missing}")
    return True


def extract_target(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Extract target variable (forward return).

    Args:
        df: DataFrame with close prices or return column.
        horizon: Prediction horizon in periods.

    Returns:
        Series with forward returns.

    Raises:
        FeatureExtractionError: If target cannot be computed.
    """
    if "return" in df.columns:
        # Use forward return
        target = df["return"].shift(-horizon)
    elif "close" in df.columns:
        # Compute forward return from close prices
        target = np.log(df["close"].shift(-horizon) / df["close"])
    else:
        raise FeatureExtractionError("Cannot compute target: need 'return' or 'close' column")

    return target.rename(f"target_h{horizon}")


def calculate_baseline_volatility(
    df: pd.DataFrame,
    column: str = "return",
    window: int = 365,
) -> float:
    """Calculate baseline volatility for threshold scaling.

    Uses the median of rolling volatility over the dataset to establish
    a "normal" volatility level. This baseline is used for volatility-adaptive
    signal threshold scaling.

    Args:
        df: DataFrame with return column.
        column: Column to calculate volatility from (default: "return").
        window: Rolling window for std calculation (default: 365 for ~1 year).

    Returns:
        Baseline volatility value (median of rolling std).

    Raises:
        FeatureExtractionError: If the specified column is not found.
    """
    if column not in df.columns:
        raise FeatureExtractionError(f"Column '{column}' not found in DataFrame")

    # Use a window that's at most the length of the data
    effective_window = min(window, len(df))
    if effective_window < 2:
        return 0.02  # Default fallback for insufficient data

    rolling_std = df[column].rolling(window=effective_window).std()
    baseline = rolling_std.median()

    # Return the median, or a sensible default if NaN
    return float(baseline) if pd.notna(baseline) else 0.02


def extract_features_hourly(
    df: pd.DataFrame,
    drop_na: bool = True,
    daily_equivalent: bool = True,
) -> pd.DataFrame:
    """Extract features from hourly candle data.

    Adapts rolling windows to hourly frequency while maintaining
    approximately the same time coverage as daily features.

    Args:
        df: DataFrame with OHLCV columns (hourly candles).
        drop_na: If True, remove rows with NaN values from rolling windows.
        daily_equivalent: If True, use windows that match daily periods
                         (168h for 7 days, 720h for 30 days).

    Returns:
        DataFrame with original columns plus 8 feature columns.

    Raises:
        FeatureExtractionError: If required columns are missing or data is empty.
    """
    if df.empty:
        raise FeatureExtractionError("DataFrame is empty")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise FeatureExtractionError(f"Missing required columns: {missing}")

    logger.info("Extracting hourly features", num_rows=len(df))

    result = df.copy()

    # Base features (same formulas as daily)
    result["return"] = np.log(result["close"] / result["close"].shift(1))
    result["range"] = np.log(result["high"] / result["low"])
    result["body"] = np.log(result["close"] / result["open"])

    # Handle zero volume
    volume = result["volume"].replace(0, 1e-10)
    result["dlog_volume"] = np.log(volume) - np.log(volume.shift(1))

    # Rolling stats with hourly-adjusted windows
    if daily_equivalent:
        window_7 = HOURLY_ROLLING_WINDOW_7  # 168 hours = 7 days
        window_30 = HOURLY_ROLLING_WINDOW_30  # 720 hours = 30 days
    else:
        window_7 = ROLLING_WINDOW_7  # 7 periods
        window_30 = ROLLING_WINDOW_30  # 30 periods

    result["ret_mean_7"] = result["return"].rolling(window=window_7).mean()
    result["ret_std_7"] = result["return"].rolling(window=window_7).std()
    result["ret_mean_30"] = result["return"].rolling(window=window_30).mean()
    result["ret_std_30"] = result["return"].rolling(window=window_30).std()

    if drop_na:
        initial_count = len(result)
        result = result.dropna().reset_index(drop=True)
        dropped = initial_count - len(result)
        logger.info(
            "Hourly features extracted",
            num_rows=len(result),
            num_features=len(FEATURE_COLUMNS),
            dropped_rows=dropped,
        )
    else:
        logger.info(
            "Hourly features extracted (with NaN)",
            num_rows=len(result),
            num_features=len(FEATURE_COLUMNS),
        )

    return result
