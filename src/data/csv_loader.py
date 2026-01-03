"""CSV data loader for Bitcoin Trading Model.

Loads and validates OHLCV data from KuCoin/Kraken CSV format.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Expected columns in KuCoin/Kraken CSV formats
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["trades", "turnover"]


class CSVLoadError(Exception):
    """Error loading or validating CSV data."""

    pass


def load_kraken_csv(file_path: str | Path) -> pd.DataFrame:
    """Load Kraken CSV file with validation.

    Args:
        file_path: Path to CSV file.

    Returns:
        DataFrame with validated OHLCV data.

    Raises:
        CSVLoadError: If file is invalid or fails validation.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise CSVLoadError(f"File not found: {file_path}")

    logger.info("Loading CSV file", file_path=str(file_path))

    try:
        # Try comma-separated first (standard CSV)
        df = pd.read_csv(file_path)

        # If only one column, try tab-separated
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep="\t")

    except Exception as e:
        raise CSVLoadError(f"Failed to read CSV file: {e}") from e

    # Validate required columns
    _validate_columns(df)

    # Convert timestamp to datetime
    df = _convert_timestamp(df)

    # Validate price consistency
    _validate_prices(df)

    # Sort by timestamp and reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "CSV loaded successfully",
        rows=len(df),
        date_range=f"{df['timestamp'].min()} to {df['timestamp'].max()}",
    )

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Validate required columns are present.

    Args:
        df: DataFrame to validate.

    Raises:
        CSVLoadError: If required columns are missing.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise CSVLoadError(f"Missing required columns: {missing}")


def _convert_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp column to datetime.

    Args:
        df: DataFrame with timestamp column.

    Returns:
        DataFrame with datetime timestamp.

    Raises:
        CSVLoadError: If timestamp conversion fails.
    """
    df = df.copy()

    try:
        # Check if timestamp is numeric (Unix timestamp)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    except Exception as e:
        raise CSVLoadError(f"Failed to convert timestamp: {e}") from e

    return df


def _validate_prices(df: pd.DataFrame) -> None:
    """Validate price consistency.

    Checks:
    - high >= low
    - close within [low, high]
    - open within [low, high]
    - All prices positive
    - All volumes non-negative

    Args:
        df: DataFrame with OHLCV columns.

    Raises:
        CSVLoadError: If validation fails.
    """
    # Check high >= low
    invalid_range = df[df["high"] < df["low"]]
    if not invalid_range.empty:
        raise CSVLoadError(
            f"Invalid data: high < low detected at rows: {invalid_range.index.tolist()[:5]}"
        )

    # Check close within [low, high]
    invalid_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
    if not invalid_close.empty:
        raise CSVLoadError(
            f"Invalid data: close outside [low, high] at rows: {invalid_close.index.tolist()[:5]}"
        )

    # Check open within [low, high]
    invalid_open = df[(df["open"] < df["low"]) | (df["open"] > df["high"])]
    if not invalid_open.empty:
        raise CSVLoadError(
            f"Invalid data: open outside [low, high] at rows: {invalid_open.index.tolist()[:5]}"
        )

    # Check positive prices
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            raise CSVLoadError(f"Invalid data: non-positive {col} detected")

    # Check non-negative volume
    if (df["volume"] < 0).any():
        raise CSVLoadError("Invalid data: negative volume detected")


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate a DataFrame has required OHLCV structure.

    Args:
        df: DataFrame to validate.

    Returns:
        True if valid.

    Raises:
        CSVLoadError: If validation fails.
    """
    _validate_columns(df)
    _validate_prices(df)
    return True
