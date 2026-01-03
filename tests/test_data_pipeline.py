"""Tests for data pipeline components."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.csv_loader import CSVLoadError, load_kraken_csv, validate_dataframe
from src.data.feature_extractor import (
    FEATURE_COLUMNS,
    FeatureExtractionError,
    extract_features,
    extract_target,
    get_feature_columns,
    validate_features,
)
from src.data.normalizer import (
    NormalizationError,
    RollingNormalizer,
    normalize_features,
)
from src.data.pipeline import DataPipeline, PipelineError, prepare_training_data
from src.data.kucoin_client import KuCoinCandle
from src.database.operations import count_candles


class TestCSVLoader:
    """Test CSV loader functionality."""

    def test_load_valid_csv(self, sample_csv_file):
        """Test loading a valid CSV file."""
        df = load_kraken_csv(sample_csv_file)

        assert len(df) == 50
        assert "timestamp" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_nonexistent_file(self, tmp_path):
        """Test error for nonexistent file."""
        with pytest.raises(CSVLoadError, match="File not found"):
            load_kraken_csv(tmp_path / "nonexistent.csv")

    def test_load_missing_columns(self, tmp_path):
        """Test error for missing required columns."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text("timestamp,open,high,low\n1,2,3,4")

        with pytest.raises(CSVLoadError, match="Missing required columns"):
            load_kraken_csv(csv_file)

    def test_load_invalid_prices(self, tmp_path):
        """Test error for invalid price data (high < low)."""
        csv_file = tmp_path / "invalid_prices.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "1381017600,100,90,95,95,10"  # high < low
        )

        with pytest.raises(CSVLoadError, match="high < low"):
            load_kraken_csv(csv_file)

    def test_load_negative_prices(self, tmp_path):
        """Test error for negative prices."""
        csv_file = tmp_path / "negative_prices.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "1381017600,-100,-90,-110,-95,10"  # All negative
        )

        with pytest.raises(CSVLoadError, match="non-positive"):
            load_kraken_csv(csv_file)

    def test_load_close_outside_range(self, tmp_path):
        """Test error for close price outside [low, high]."""
        csv_file = tmp_path / "invalid_close.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "1381017600,100,110,90,120,10"  # close > high
        )

        with pytest.raises(CSVLoadError, match="close outside"):
            load_kraken_csv(csv_file)

    def test_load_with_turnover_column(self, tmp_path):
        """Accept KuCoin turnover column."""
        csv_file = tmp_path / "kucoin.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume,turnover\n"
            "1381017600,100,110,90,105,10,1000\n"
            "1381104000,105,115,95,110,12,1100\n"
        )

        df = load_kraken_csv(csv_file)
        assert "turnover" in df.columns
        assert len(df) == 2

    def test_load_negative_volume(self, tmp_path):
        """Test error for negative volume."""
        csv_file = tmp_path / "negative_volume.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "1381017600,100,110,90,100,-10"
        )

        with pytest.raises(CSVLoadError, match="negative volume"):
            load_kraken_csv(csv_file)

    def test_validate_dataframe(self, sample_dataframe):
        """Test DataFrame validation."""
        assert validate_dataframe(sample_dataframe) is True

    def test_validate_invalid_dataframe(self):
        """Test validation fails for invalid DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(CSVLoadError, match="Missing required columns"):
            validate_dataframe(df)


class TestFeatureExtractor:
    """Test feature extraction functionality."""

    def test_extract_features(self, sample_dataframe):
        """Test extracting features from OHLCV data."""
        result = extract_features(sample_dataframe)

        # Check all feature columns present
        for col in FEATURE_COLUMNS:
            assert col in result.columns

        # Check no NaN values (after dropping)
        for col in FEATURE_COLUMNS:
            assert not result[col].isna().any()

    def test_extract_features_empty_df(self):
        """Test error for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(FeatureExtractionError, match="empty"):
            extract_features(df)

    def test_extract_features_missing_columns(self):
        """Test error for missing OHLCV columns."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(FeatureExtractionError, match="Missing required"):
            extract_features(df)

    def test_extract_features_return_calculation(self, sample_dataframe):
        """Test return calculation is correct."""
        result = extract_features(sample_dataframe, drop_na=False)

        # Manual calculation for first valid return
        expected_return = np.log(sample_dataframe["close"].iloc[1] / sample_dataframe["close"].iloc[0])
        actual_return = result["return"].iloc[1]

        assert np.isclose(actual_return, expected_return)

    def test_extract_features_range_calculation(self, sample_dataframe):
        """Test range calculation is correct."""
        result = extract_features(sample_dataframe, drop_na=False)

        # Manual calculation
        expected_range = np.log(sample_dataframe["high"].iloc[0] / sample_dataframe["low"].iloc[0])
        actual_range = result["range"].iloc[0]

        assert np.isclose(actual_range, expected_range)

    def test_extract_features_body_calculation(self, sample_dataframe):
        """Test body calculation is correct."""
        result = extract_features(sample_dataframe, drop_na=False)

        # Manual calculation
        expected_body = np.log(sample_dataframe["close"].iloc[0] / sample_dataframe["open"].iloc[0])
        actual_body = result["body"].iloc[0]

        assert np.isclose(actual_body, expected_body)

    def test_extract_features_preserves_original_columns(self, sample_dataframe):
        """Test original columns are preserved."""
        result = extract_features(sample_dataframe)

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_get_feature_columns(self):
        """Test getting feature column list."""
        columns = get_feature_columns()

        assert len(columns) == 8
        assert "return" in columns
        assert "ret_std_30" in columns

    def test_validate_features(self, sample_dataframe):
        """Test feature validation."""
        features = extract_features(sample_dataframe)
        assert validate_features(features) is True

    def test_validate_features_missing(self):
        """Test validation fails for missing features."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(FeatureExtractionError, match="Missing feature"):
            validate_features(df)

    def test_extract_target(self, sample_dataframe):
        """Test target extraction."""
        features = extract_features(sample_dataframe, drop_na=False)
        target = extract_target(features, horizon=1)

        assert target.name == "target_h1"
        # Target at t should equal return at t+1
        # (shifted forward, so we compare offset)

    def test_extract_target_from_close(self, sample_dataframe):
        """Test target extraction from close prices."""
        target = extract_target(sample_dataframe, horizon=1)

        assert target.name == "target_h1"
        assert len(target) == len(sample_dataframe)


class TestNormalizer:
    """Test normalization functionality."""

    def test_rolling_normalizer_fit(self, sample_dataframe):
        """Test fitting normalizer."""
        features = extract_features(sample_dataframe)
        normalizer = RollingNormalizer(window=7)

        normalizer.fit(features, columns=["return", "range"])

        assert normalizer.is_fitted
        assert "return" in normalizer.feature_stats
        assert "range" in normalizer.feature_stats

    def test_rolling_normalizer_transform(self, sample_dataframe):
        """Test transforming data with normalizer."""
        features = extract_features(sample_dataframe)
        normalizer = RollingNormalizer(window=7)

        result = normalizer.transform(features, columns=["return"])

        # Normalized values should have different distribution
        assert "return" in result.columns

    def test_rolling_normalizer_fit_transform(self, sample_dataframe):
        """Test fit_transform in one step."""
        features = extract_features(sample_dataframe)
        normalizer = RollingNormalizer(window=7)

        result = normalizer.fit_transform(features, columns=["return"])

        assert normalizer.is_fitted
        assert "return" in result.columns

    def test_rolling_normalizer_transform_single(self, sample_dataframe):
        """Test transforming single observation."""
        features = extract_features(sample_dataframe)
        normalizer = RollingNormalizer(window=7)
        normalizer.fit(features, columns=["return", "range"])

        single = {"return": 0.01, "range": 0.02}
        result = normalizer.transform_single(single)

        assert "return" in result
        assert "range" in result
        assert isinstance(result["return"], float)

    def test_rolling_normalizer_transform_single_not_fitted(self):
        """Test error when transforming without fitting."""
        normalizer = RollingNormalizer()

        with pytest.raises(NormalizationError, match="not fitted"):
            normalizer.transform_single({"return": 0.01})

    def test_rolling_normalizer_save_load(self, sample_dataframe, tmp_path):
        """Test saving and loading normalizer."""
        features = extract_features(sample_dataframe)
        normalizer = RollingNormalizer(window=7)
        normalizer.fit(features, columns=["return"])

        # Save
        save_path = tmp_path / "normalizer.pkl"
        normalizer.save(save_path)

        # Load
        loaded = RollingNormalizer.load(save_path)

        assert loaded.is_fitted
        assert loaded.window == normalizer.window
        assert loaded.feature_stats == normalizer.feature_stats

    def test_normalize_features_convenience(self, sample_dataframe):
        """Test convenience function for normalization."""
        features = extract_features(sample_dataframe)
        result = normalize_features(features, columns=["return", "range"])

        assert "return" in result.columns
        assert "range" in result.columns


class TestDataPipeline:
    """Test end-to-end data pipeline."""

    def test_load_csv_to_database(self, session, sample_csv_file):
        """Test loading CSV to database."""
        pipeline = DataPipeline(session)
        count = pipeline.load_csv_to_database(sample_csv_file)

        assert count == 50

    def test_load_csv_skip_duplicates(self, session, sample_csv_file):
        """Test skipping duplicate candles."""
        pipeline = DataPipeline(session)

        # Load once
        count1 = pipeline.load_csv_to_database(sample_csv_file)
        # Load again - should skip all
        count2 = pipeline.load_csv_to_database(sample_csv_file)

        assert count1 == 50
        assert count2 == 0  # All duplicates skipped

    def test_load_csv_replace_existing(self, session, sample_csv_file):
        """Test replacing existing data."""
        pipeline = DataPipeline(session)

        # Load twice with replace
        pipeline.load_csv_to_database(sample_csv_file)
        count = pipeline.load_csv_to_database(sample_csv_file, replace_existing=True)

        assert count == 50

    def test_extract_and_store_features(self, session, sample_csv_file):
        """Test feature extraction and storage."""
        pipeline = DataPipeline(session)

        # Load candles first
        pipeline.load_csv_to_database(sample_csv_file)

        # Extract features
        feature_count = pipeline.extract_and_store_features()

        # Should have fewer features due to rolling window
        assert feature_count > 0
        assert feature_count < 50  # Some rows dropped for rolling windows

    def test_extract_features_no_candles(self, session):
        """Test error when no candles in database."""
        pipeline = DataPipeline(session)

        with pytest.raises(PipelineError, match="No candles"):
            pipeline.extract_and_store_features()

    def test_load_kucoin_to_database(self, session, monkeypatch):
        """Test loading KuCoin candles into database."""
        pipeline = DataPipeline(session)

        def fake_backfill(*_args, **_kwargs):
            return [
                KuCoinCandle(100, 1.0, 1.1, 1.2, 0.9, 10.0, 20.0),
                KuCoinCandle(200, 1.1, 1.2, 1.3, 1.0, 11.0, 21.0),
            ]

        monkeypatch.setattr("src.data.pipeline.backfill_kucoin_candles", fake_backfill)

        count = pipeline.load_kucoin_to_database(
            symbol="BTC-USDT",
            timeframe="1hour",
            start_at=0,
            end_at=300,
            replace_existing=True,
        )

        assert count == 2
        assert count_candles(session) == 2

    def test_run_full_pipeline(self, session, sample_csv_file):
        """Test full pipeline execution."""
        pipeline = DataPipeline(session)
        result = pipeline.run_full_pipeline(sample_csv_file)

        assert "candles" in result
        assert "features" in result
        assert result["candles"] == 50
        assert result["features"] > 0


class TestPrepareTrainingData:
    """Test training data preparation."""

    def test_prepare_training_data_empty(self, session):
        """Test with empty database."""
        df, normalizer = prepare_training_data(session)

        assert df.empty
        assert normalizer is None

    def test_prepare_training_data_with_features(self, session, sample_csv_file):
        """Test preparing data with features."""
        # Load data first
        pipeline = DataPipeline(session)
        pipeline.run_full_pipeline(sample_csv_file)

        df, normalizer = prepare_training_data(session, normalize=True)

        assert not df.empty
        assert normalizer is not None
        assert "return" in df.columns

    def test_prepare_training_data_no_normalize(self, session, sample_csv_file):
        """Test without normalization."""
        pipeline = DataPipeline(session)
        pipeline.run_full_pipeline(sample_csv_file)

        df, normalizer = prepare_training_data(session, normalize=False)

        assert not df.empty
        assert normalizer is None
