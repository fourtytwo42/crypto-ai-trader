"""End-to-end data pipeline for Bitcoin Trading Model.

Orchestrates CSV loading, feature extraction, normalization,
and database storage.
"""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import structlog
from sqlalchemy.orm import Session

from src.data.csv_loader import load_kraken_csv
from src.data.feature_extractor import extract_features, get_feature_columns
from src.data.kucoin_client import backfill_kucoin_candles
from src.data.normalizer import RollingNormalizer
from src.database.operations import (
    create_candle,
    create_feature,
    delete_all_candles,
    get_candle_by_timestamp,
)

logger = structlog.get_logger(__name__)


class PipelineError(Exception):
    """Error in data pipeline."""

    pass


class DataPipeline:
    """End-to-end data processing pipeline."""

    def __init__(self, session: Session):
        """Initialize pipeline with database session.

        Args:
            session: SQLAlchemy database session.
        """
        self.session = session
        self.normalizer = RollingNormalizer(window=30)

    def load_csv_to_database(
        self,
        file_path: str | Path,
        replace_existing: bool = False,
    ) -> int:
        """Load CSV file and store candles in database.

        Args:
            file_path: Path to CSV file.
            replace_existing: If True, delete existing candles first.

        Returns:
            Number of candles loaded.

        Raises:
            PipelineError: If loading fails.
        """
        try:
            # Load CSV
            df = load_kraken_csv(file_path)

            if replace_existing:
                delete_all_candles(self.session)

            # Insert candles
            count = 0
            for _, row in df.iterrows():
                # Check if candle already exists
                existing = get_candle_by_timestamp(self.session, row["timestamp"])
                if existing:
                    continue

                create_candle(
                    self.session,
                    timestamp=row["timestamp"],
                    open_price=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=Decimal(str(row["volume"])),
                    trades=int(row["trades"]) if "trades" in row and pd.notna(row["trades"]) else None,
                )
                count += 1

            self.session.commit()
            logger.info("Loaded candles to database", count=count)
            return count

        except Exception as e:
            self.session.rollback()
            raise PipelineError(f"Failed to load CSV: {e}") from e

    def extract_and_store_features(self) -> int:
        """Extract features from candles and store in database.

        Returns:
            Number of features created.

        Raises:
            PipelineError: If extraction fails.
        """
        try:
            # Get all candles as DataFrame
            from src.database.operations import get_all_candles

            candles = get_all_candles(self.session)
            if not candles:
                raise PipelineError("No candles in database")

            # Convert to DataFrame
            df = pd.DataFrame([{
                "id": c.id,
                "timestamp": c.timestamp,
                "open": float(c.open),
                "high": float(c.high),
                "low": float(c.low),
                "close": float(c.close),
                "volume": float(c.volume),
            } for c in candles])

            # Extract features
            features_df = extract_features(df, drop_na=True)

            # Store features
            count = 0
            for _, row in features_df.iterrows():
                # Find candle ID by matching timestamp
                candle = get_candle_by_timestamp(self.session, row["timestamp"])
                if not candle:
                    continue

                create_feature(
                    self.session,
                    candle_id=candle.id,
                    timestamp=row["timestamp"],
                    return_=Decimal(str(row["return"])),
                    range_=Decimal(str(row["range"])),
                    body=Decimal(str(row["body"])),
                    dlog_volume=Decimal(str(row["dlog_volume"])),
                    ret_mean_7=Decimal(str(row["ret_mean_7"])) if pd.notna(row["ret_mean_7"]) else None,
                    ret_std_7=Decimal(str(row["ret_std_7"])) if pd.notna(row["ret_std_7"]) else None,
                    ret_mean_30=Decimal(str(row["ret_mean_30"])) if pd.notna(row["ret_mean_30"]) else None,
                    ret_std_30=Decimal(str(row["ret_std_30"])) if pd.notna(row["ret_std_30"]) else None,
                )
                count += 1

            self.session.commit()
            logger.info("Extracted and stored features", count=count)
            return count

        except Exception as e:
            self.session.rollback()
            raise PipelineError(f"Failed to extract features: {e}") from e

    def load_kucoin_to_database(
        self,
        symbol: str,
        timeframe: str,
        start_at: int,
        end_at: int | None = None,
        replace_existing: bool = False,
    ) -> int:
        """Backfill KuCoin candles and store in database.

        Args:
            symbol: KuCoin symbol (e.g., BTC-USDT).
            timeframe: KuCoin timeframe string (e.g., 1hour).
            start_at: Start timestamp (unix seconds).
            end_at: End timestamp (unix seconds). Defaults to now.
            replace_existing: If True, delete existing candles first.

        Returns:
            Number of candles loaded.

        Raises:
            PipelineError: If loading fails.
        """
        try:
            if replace_existing:
                delete_all_candles(self.session)

            count = 0
            for candle in backfill_kucoin_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_at=start_at,
                end_at=end_at,
            ):
                ts = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
                existing = get_candle_by_timestamp(self.session, ts)
                if existing:
                    continue

                create_candle(
                    self.session,
                    timestamp=ts,
                    open_price=Decimal(str(candle.open)),
                    high=Decimal(str(candle.high)),
                    low=Decimal(str(candle.low)),
                    close=Decimal(str(candle.close)),
                    volume=Decimal(str(candle.volume)),
                    trades=None,
                )
                count += 1

            self.session.commit()
            logger.info("Loaded KuCoin candles to database", count=count)
            return count

        except Exception as e:
            self.session.rollback()
            raise PipelineError(f"Failed to load KuCoin candles: {e}") from e

    def load_kucoin_hourly_to_database(
        self,
        symbol: str = "BTC-USDT",
        hours_back: int = 8760,  # 1 year of hourly data (365 * 24)
        replace_existing: bool = False,
    ) -> int:
        """Load hourly KuCoin data for sub-daily predictions.

        Fetches hourly candle data from KuCoin for use with shorter
        prediction horizons (6-8 hours).

        Args:
            symbol: KuCoin symbol (default: BTC-USDT).
            hours_back: Number of hours to backfill (default: 8760 = 1 year).
            replace_existing: If True, delete existing candles first.

        Returns:
            Number of candles loaded.

        Raises:
            PipelineError: If loading fails.
        """
        import time

        end_at = int(time.time())
        start_at = end_at - (hours_back * 3600)

        logger.info(
            "Loading hourly KuCoin data",
            symbol=symbol,
            hours_back=hours_back,
            start_at=start_at,
            end_at=end_at,
        )

        return self.load_kucoin_to_database(
            symbol=symbol,
            timeframe="1hour",
            start_at=start_at,
            end_at=end_at,
            replace_existing=replace_existing,
        )

    def run_full_pipeline(
        self,
        file_path: str | Path,
        replace_existing: bool = False,
    ) -> dict[str, int]:
        """Run complete data pipeline.

        Args:
            file_path: Path to CSV file.
            replace_existing: If True, replace existing data.

        Returns:
            Dictionary with counts for each step.
        """
        logger.info("Starting full pipeline", file_path=str(file_path))

        # Load candles
        candle_count = self.load_csv_to_database(file_path, replace_existing)

        # Extract features
        feature_count = self.extract_and_store_features()

        result = {
            "candles": candle_count,
            "features": feature_count,
        }

        logger.info("Pipeline complete", **result)
        return result


def prepare_training_data(
    session: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    normalize: bool = True,
) -> tuple[pd.DataFrame, RollingNormalizer | None]:
    """Prepare data for model training.

    Args:
        session: Database session.
        start_date: Start date filter.
        end_date: End date filter.
        normalize: If True, apply rolling normalization.

    Returns:
        Tuple of (features DataFrame, normalizer if normalize=True else None).
    """
    from src.database.operations import get_features_in_range, get_all_candles

    # Get features
    if start_date and end_date:
        features = get_features_in_range(session, start_date, end_date)
    else:
        # Get all candles and features
        candles = get_all_candles(session)
        # Convert to dataframe format expected by training
        features = []
        for candle in candles:
            if candle.features:
                for f in candle.features:
                    features.append(f)

    if not features:
        return pd.DataFrame(), None

    # Convert to DataFrame
    df = pd.DataFrame([{
        "timestamp": f.timestamp,
        "close": float(f.candle.close) if f.candle else None,
        "return": float(f.return_) if f.return_ else None,
        "range": float(f.range) if f.range else None,
        "body": float(f.body) if f.body else None,
        "dlog_volume": float(f.dlog_volume) if f.dlog_volume else None,
        "ret_mean_7": float(f.ret_mean_7) if f.ret_mean_7 else None,
        "ret_std_7": float(f.ret_std_7) if f.ret_std_7 else None,
        "ret_mean_30": float(f.ret_mean_30) if f.ret_mean_30 else None,
        "ret_std_30": float(f.ret_std_30) if f.ret_std_30 else None,
    } for f in features])

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    normalizer = None
    if normalize:
        normalizer = RollingNormalizer(window=30)
        feature_cols = get_feature_columns()
        df = normalizer.fit_transform(df, feature_cols)

    return df, normalizer
