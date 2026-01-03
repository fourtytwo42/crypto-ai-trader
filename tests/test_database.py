"""Tests for database connection and models."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.database.models import (
    Backtest,
    BacktestTrade,
    Base,
    Candle,
    Feature,
    Model,
    Prediction,
)
from src.database.operations import (
    count_candles,
    create_backtest,
    create_backtest_trade,
    create_candle,
    create_feature,
    create_model,
    create_prediction,
    delete_all_candles,
    get_all_backtests,
    get_all_candles,
    get_all_models,
    get_backtest_by_id,
    get_backtests_by_model,
    get_candle_by_timestamp,
    get_candles_in_range,
    get_features_in_range,
    get_latest_candles,
    get_latest_features,
    get_latest_prediction,
    get_model_by_id,
    get_model_by_name,
    get_predictions_by_model,
    get_trades_by_backtest,
    update_model_metrics,
)


class TestDatabaseConnection:
    """Test database connection and session management."""

    def test_tables_created(self, engine):
        """Test all tables are created correctly."""
        # Check tables exist
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())

        expected_tables = {
            "candles", "features", "models",
            "predictions", "backtests", "backtest_trades"
        }
        assert expected_tables.issubset(tables)

    def test_session_rollback_on_error(self, session):
        """Test session rolls back on error."""
        # Create valid candle
        candle = Candle(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        session.add(candle)
        session.flush()

        # Try to create duplicate (should fail)
        duplicate = Candle(
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )
        session.add(duplicate)

        with pytest.raises(IntegrityError):
            session.flush()


class TestCandleModel:
    """Test Candle database model."""

    def test_create_candle(self, session):
        """Test creating a candle record."""
        candle = create_candle(
            session,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
            open_price=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
            trades=50,
        )

        assert candle.id is not None
        assert candle.open == Decimal("100")
        assert candle.close == Decimal("105")
        assert candle.trades == 50

    def test_candle_unique_timestamp(self, session):
        """Test candle timestamp is unique."""
        ts = datetime(2020, 1, 1, tzinfo=timezone.utc)

        create_candle(
            session,
            timestamp=ts,
            open_price=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )

        with pytest.raises(IntegrityError):
            create_candle(
                session,
                timestamp=ts,  # Duplicate timestamp
                open_price=Decimal("101"),
                high=Decimal("111"),
                low=Decimal("91"),
                close=Decimal("106"),
                volume=Decimal("1001"),
            )

    def test_get_candle_by_timestamp(self, session, sample_candles):
        """Test getting candle by timestamp."""
        ts = sample_candles[0].timestamp
        candle = get_candle_by_timestamp(session, ts)

        assert candle is not None
        assert candle.timestamp == ts

    def test_get_candles_in_range(self, session, sample_candles):
        """Test getting candles in date range."""
        start = sample_candles[5].timestamp
        end = sample_candles[15].timestamp

        candles = get_candles_in_range(session, start, end)

        assert len(candles) == 11
        assert candles[0].timestamp == start
        assert candles[-1].timestamp == end

    def test_get_latest_candles(self, session, sample_candles):
        """Test getting latest candles."""
        candles = get_latest_candles(session, limit=10)

        assert len(candles) == 10
        # Should be in descending order
        assert candles[0].timestamp > candles[-1].timestamp

    def test_get_all_candles(self, session, sample_candles):
        """Test getting all candles."""
        candles = get_all_candles(session)
        assert len(candles) == len(sample_candles)

    def test_count_candles(self, session, sample_candles):
        """Test counting candles."""
        count = count_candles(session)
        assert count == len(sample_candles)

    def test_delete_all_candles(self, session, sample_candles):
        """Test deleting all candles."""
        count = delete_all_candles(session)
        assert count == len(sample_candles)
        assert count_candles(session) == 0


class TestFeatureModel:
    """Test Feature database model."""

    def test_create_feature(self, session, sample_candles):
        """Test creating a feature record."""
        candle = sample_candles[0]
        feature = create_feature(
            session,
            candle_id=candle.id,
            timestamp=candle.timestamp,
            return_=Decimal("0.01"),
            range_=Decimal("0.02"),
            body=Decimal("0.005"),
            dlog_volume=Decimal("0.1"),
            ret_mean_7=Decimal("0.005"),
            ret_std_7=Decimal("0.01"),
            ret_mean_30=Decimal("0.003"),
            ret_std_30=Decimal("0.015"),
        )

        assert feature.id is not None
        assert feature.candle_id == candle.id
        assert feature.return_ == Decimal("0.01")
        assert feature.range == Decimal("0.02")

    def test_feature_cascade_delete(self, session, sample_candles):
        """Test features are deleted when candle is deleted."""
        candle = sample_candles[0]
        feature = create_feature(
            session,
            candle_id=candle.id,
            timestamp=candle.timestamp,
            return_=Decimal("0.01"),
        )

        feature_id = feature.id
        session.delete(candle)
        session.flush()

        # Feature should be deleted
        from sqlalchemy import select
        stmt = select(Feature).where(Feature.id == feature_id)
        result = session.execute(stmt).scalar_one_or_none()
        assert result is None

    def test_get_features_in_range(self, session, sample_features):
        """Test getting features in date range."""
        start = sample_features[5].timestamp
        end = sample_features[15].timestamp

        features = get_features_in_range(session, start, end)

        assert len(features) >= 10  # At least 10 features
        assert features[0].timestamp >= start
        assert features[-1].timestamp <= end

    def test_get_latest_features(self, session, sample_features):
        """Test getting latest features."""
        features = get_latest_features(session, limit=5)

        assert len(features) == 5
        assert features[0].timestamp > features[-1].timestamp


class TestModelModel:
    """Test Model database model."""

    def test_create_model(self, session):
        """Test creating a model record."""
        model = create_model(
            session,
            name="test_model",
            model_type="patchtst",
            version="1.0.0",
            file_path="models/test.pt",
            config={"context_length": 128, "hidden_size": 512},
            metrics={"mae": 0.01},
        )

        assert model.id is not None
        assert model.name == "test_model"
        assert model.config["context_length"] == 128

    def test_model_unique_name(self, session):
        """Test model name is unique."""
        create_model(
            session,
            name="unique_model",
            model_type="patchtst",
            version="1.0.0",
            file_path="models/test1.pt",
            config={},
        )

        with pytest.raises(IntegrityError):
            create_model(
                session,
                name="unique_model",  # Duplicate name
                model_type="nhits",
                version="2.0.0",
                file_path="models/test2.pt",
                config={},
            )

    def test_get_model_by_id(self, session, sample_model):
        """Test getting model by ID."""
        model = get_model_by_id(session, sample_model.id)
        assert model is not None
        assert model.name == sample_model.name

    def test_get_model_by_name(self, session, sample_model):
        """Test getting model by name."""
        model = get_model_by_name(session, sample_model.name)
        assert model is not None
        assert model.id == sample_model.id

    def test_get_all_models(self, session, sample_model):
        """Test getting all models."""
        models = get_all_models(session)
        assert len(models) >= 1
        assert any(m.id == sample_model.id for m in models)

    def test_update_model_metrics(self, session, sample_model):
        """Test updating model metrics."""
        new_metrics = {"mae": 0.005, "mape": 1.0}
        updated = update_model_metrics(session, sample_model.id, new_metrics)

        assert updated is not None
        assert updated.metrics == new_metrics


class TestPredictionModel:
    """Test Prediction database model."""

    def test_create_prediction(self, session, sample_model):
        """Test creating a prediction record."""
        prediction = create_prediction(
            session,
            model_id=sample_model.id,
            timestamp=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
            prediction_timestamp=datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc),
            q10=Decimal("-0.02"),
            q50=Decimal("0.01"),
            q90=Decimal("0.04"),
            signal="long",
            threshold=Decimal("0.005"),
        )

        assert prediction.id is not None
        assert prediction.signal == "long"
        assert prediction.q50 == Decimal("0.01")

    def test_prediction_cascade_delete(self, session, sample_model):
        """Test predictions are deleted when model is deleted."""
        prediction = create_prediction(
            session,
            model_id=sample_model.id,
            timestamp=datetime(2023, 6, 1, 12, 0, tzinfo=timezone.utc),
            prediction_timestamp=datetime(2023, 6, 2, 0, 0, tzinfo=timezone.utc),
            signal="flat",
        )

        prediction_id = prediction.id
        session.delete(sample_model)
        session.flush()

        # Prediction should be deleted
        from sqlalchemy import select
        stmt = select(Prediction).where(Prediction.id == prediction_id)
        result = session.execute(stmt).scalar_one_or_none()
        assert result is None

    def test_get_predictions_by_model(self, session, sample_predictions):
        """Test getting predictions by model."""
        model_id = sample_predictions[0].model_id
        predictions = get_predictions_by_model(session, model_id)

        assert len(predictions) == len(sample_predictions)

    def test_get_latest_prediction(self, session, sample_predictions):
        """Test getting latest prediction."""
        model_id = sample_predictions[0].model_id
        prediction = get_latest_prediction(session, model_id)

        assert prediction is not None
        # Should be the most recent
        assert prediction.prediction_timestamp == max(
            p.prediction_timestamp for p in sample_predictions
        )


class TestBacktestModel:
    """Test Backtest database model."""

    def test_create_backtest(self, session, sample_model):
        """Test creating a backtest record."""
        backtest = create_backtest(
            session,
            model_id=sample_model.id,
            name="test_backtest",
            start_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
            metrics={"sharpe_ratio": 1.2, "win_rate": 0.55},
            fees=Decimal("0.001"),
            slippage=Decimal("0.0005"),
        )

        assert backtest.id is not None
        assert backtest.name == "test_backtest"
        assert backtest.metrics["sharpe_ratio"] == 1.2

    def test_get_backtest_by_id(self, session, sample_backtest):
        """Test getting backtest by ID."""
        backtest = get_backtest_by_id(session, sample_backtest.id)
        assert backtest is not None
        assert backtest.name == sample_backtest.name

    def test_get_backtests_by_model(self, session, sample_backtest):
        """Test getting backtests by model."""
        backtests = get_backtests_by_model(session, sample_backtest.model_id)
        assert len(backtests) >= 1
        assert any(b.id == sample_backtest.id for b in backtests)

    def test_get_all_backtests(self, session, sample_backtest):
        """Test getting all backtests."""
        backtests = get_all_backtests(session)
        assert len(backtests) >= 1


class TestBacktestTradeModel:
    """Test BacktestTrade database model."""

    def test_create_backtest_trade(self, session, sample_backtest):
        """Test creating a backtest trade record."""
        trade = create_backtest_trade(
            session,
            backtest_id=sample_backtest.id,
            entry_timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),
            exit_timestamp=datetime(2022, 1, 2, tzinfo=timezone.utc),
            signal="long",
            entry_price=Decimal("42000"),
            exit_price=Decimal("42500"),
            pnl=Decimal("500"),
            fees=Decimal("84.5"),
            slippage=Decimal("21"),
            net_pnl=Decimal("394.5"),
        )

        assert trade.id is not None
        assert trade.signal == "long"
        assert trade.net_pnl == Decimal("394.5")

    def test_trade_cascade_delete(self, session, sample_backtest):
        """Test trades are deleted when backtest is deleted."""
        trade = create_backtest_trade(
            session,
            backtest_id=sample_backtest.id,
            entry_timestamp=datetime(2022, 6, 1, tzinfo=timezone.utc),
            exit_timestamp=datetime(2022, 6, 2, tzinfo=timezone.utc),
            signal="short",
            entry_price=Decimal("30000"),
            exit_price=Decimal("29500"),
            pnl=Decimal("500"),
            fees=Decimal("59.5"),
            slippage=Decimal("15"),
            net_pnl=Decimal("425.5"),
        )

        trade_id = trade.id
        session.delete(sample_backtest)
        session.flush()

        # Trade should be deleted
        from sqlalchemy import select
        stmt = select(BacktestTrade).where(BacktestTrade.id == trade_id)
        result = session.execute(stmt).scalar_one_or_none()
        assert result is None

    def test_get_trades_by_backtest(self, session, sample_trades):
        """Test getting trades by backtest."""
        backtest_id = sample_trades[0].backtest_id
        trades = get_trades_by_backtest(session, backtest_id)

        assert len(trades) == len(sample_trades)
        # Should be in chronological order
        for i in range(len(trades) - 1):
            assert trades[i].entry_timestamp <= trades[i + 1].entry_timestamp
