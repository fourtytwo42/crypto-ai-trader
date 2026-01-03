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

# Rolling window sizes
ROLLING_WINDOW_7 = 7
ROLLING_WINDOW_30 = 30


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
