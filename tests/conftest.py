"""Pytest configuration and fixtures.

Provides database fixtures, sample data, and test utilities
for all test modules.
"""

import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Generator

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings
from src.database.models import Base, Backtest, BacktestTrade, Candle, Feature, Model, Prediction


import uuid
from sqlalchemy.engine.url import make_url

# Use PostgreSQL for testing
DEFAULT_TEST_DB_URL = "postgresql://trading_user:password@localhost:5432/bitcoin_trading"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DB_URL)


@pytest.fixture(scope="session")
def db_schema(test_db_url: str) -> str:
    schema_name = f"test_{uuid.uuid4().hex}"
    engine = create_engine(test_db_url)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema_name}"'))
    engine.dispose()
    return schema_name


@pytest.fixture(scope="session")
def db_connect_args(db_schema: str) -> dict[str, str]:
    return {"options": f"-c search_path={db_schema}"}


@pytest.fixture(scope="session")
def test_db_url_with_schema(test_db_url: str, db_schema: str) -> str:
    url = make_url(test_db_url)
    options = f"-c search_path={db_schema}"
    return url.set(query={"options": options}).render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def test_settings(test_db_url_with_schema: str) -> Settings:
    """Create test settings."""
    return Settings(
        database_url=test_db_url_with_schema,
        model_dir="tests/test_models",
        data_dir="tests/test_data",
        train_device="cpu",
        log_level="DEBUG",
    )


@pytest.fixture(scope="session")
def engine(test_db_url: str, db_schema: str):
    """Create test database engine."""
    engine = create_engine(
        test_db_url,
        connect_args={"options": f"-c search_path={db_schema}"},
        echo=False,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    admin_engine = create_engine(test_db_url)
    with admin_engine.connect() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{db_schema}" CASCADE'))
    admin_engine.dispose()
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_database(engine):
    """Truncate tables between tests."""
    with engine.connect() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'TRUNCATE TABLE "{table.name}" RESTART IDENTITY CASCADE'))
        conn.execute(text("COMMIT"))


@pytest.fixture(scope="function")
def session(engine) -> Generator[Session, None, None]:
    """Create test database session."""
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def sample_candle_data() -> list[dict]:
    """Generate sample candle data for testing."""
    from datetime import timedelta

    base_time = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = []
    price = Decimal("10000.0")

    for i in range(50):
        # Generate realistic price movements
        change = Decimal(str((i % 7 - 3) * 50))
        price = price + change
        high = price + Decimal("100")
        low = price - Decimal("100")

        data.append({
            "timestamp": base_time + timedelta(days=i),
            "open": price - Decimal("20"),
            "high": high,
            "low": low,
            "close": price,
            "volume": Decimal(str(100 + i * 10)),
            "trades": 50 + i,
        })

    return data


@pytest.fixture
def sample_candles(session: Session, sample_candle_data: list[dict]) -> list[Candle]:
    """Create sample candles in database."""
    candles = []
    for data in sample_candle_data:
        candle = Candle(**data)
        session.add(candle)
        candles.append(candle)
    session.flush()
    return candles


@pytest.fixture
def sample_features(session: Session, sample_candles: list[Candle]) -> list[Feature]:
    """Create sample features in database."""
    features = []
    for i, candle in enumerate(sample_candles[1:], start=1):  # Skip first (no return)
        feature = Feature(
            candle_id=candle.id,
            timestamp=candle.timestamp,
            return_=Decimal("0.001") * i,
            range=Decimal("0.02"),
            body=Decimal("0.005"),
            dlog_volume=Decimal("0.1"),
            ret_mean_7=Decimal("0.0005") if i >= 7 else None,
            ret_std_7=Decimal("0.01") if i >= 7 else None,
            ret_mean_30=Decimal("0.0003") if i >= 30 else None,
            ret_std_30=Decimal("0.015") if i >= 30 else None,
        )
        session.add(feature)
        features.append(feature)
    session.flush()
    return features


@pytest.fixture
def sample_model(session: Session) -> Model:
    """Create sample model in database."""
    model = Model(
        name="test_model_v1",
        model_type="patchtst",
        version="1.0.0",
        file_path="models/test_model_v1.pt",
        scaler_path="models/test_model_v1_scaler.pkl",
        config={
            "context_length": 128,
            "horizon": 1,
            "hidden_size": 512,
            "num_layers": 6,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "epochs": 100,
            "quantiles": [0.1, 0.5, 0.9],
        },
        train_start=datetime(2013, 10, 6, tzinfo=timezone.utc),
        train_end=datetime(2019, 12, 31, tzinfo=timezone.utc),
        val_start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        val_end=datetime(2021, 12, 31, tzinfo=timezone.utc),
        test_start=datetime(2022, 1, 1, tzinfo=timezone.utc),
        test_end=datetime(2023, 12, 31, tzinfo=timezone.utc),
        metrics={
            "val_mae": 0.012,
            "val_mape": 1.5,
            "test_mae": 0.015,
            "test_mape": 2.0,
        },
    )
    session.add(model)
    session.flush()
    return model


@pytest.fixture
def sample_predictions(session: Session, sample_model: Model) -> list[Prediction]:
    """Create sample predictions in database."""
    predictions = []
    base_time = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(10):
        prediction = Prediction(
            model_id=sample_model.id,
            timestamp=datetime(2023, 1, i + 1, 12, 0, 0, tzinfo=timezone.utc),
            prediction_timestamp=datetime(2023, 1, i + 2, 0, 0, 0, tzinfo=timezone.utc),
            q10=Decimal("-0.02"),
            q50=Decimal("0.01"),
            q90=Decimal("0.04"),
            signal="long" if i % 3 == 0 else ("short" if i % 3 == 1 else "flat"),
            threshold=Decimal("0.005"),
        )
        session.add(prediction)
        predictions.append(prediction)
    session.flush()
    return predictions


@pytest.fixture
def sample_backtest(session: Session, sample_model: Model) -> Backtest:
    """Create sample backtest in database."""
    backtest = Backtest(
        model_id=sample_model.id,
        name="backtest_2023",
        start_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        train_window=1825,
        test_window=365,
        fees=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        metrics={
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "win_rate": 0.58,
            "avg_win": 0.02,
            "avg_loss": -0.015,
            "max_drawdown": -0.08,
            "num_trades": 150,
            "turnover": 0.5,
        },
    )
    session.add(backtest)
    session.flush()
    return backtest


@pytest.fixture
def sample_trades(session: Session, sample_backtest: Backtest) -> list[BacktestTrade]:
    """Create sample backtest trades in database."""
    trades = []

    for i in range(5):
        trade = BacktestTrade(
            backtest_id=sample_backtest.id,
            entry_timestamp=datetime(2022, 1, 1 + i * 7, tzinfo=timezone.utc),
            exit_timestamp=datetime(2022, 1, 2 + i * 7, tzinfo=timezone.utc),
            signal="long" if i % 2 == 0 else "short",
            entry_price=Decimal("42000") + Decimal(str(i * 100)),
            exit_price=Decimal("42500") + Decimal(str(i * 100)),
            pnl=Decimal("500") if i % 2 == 0 else Decimal("-300"),
            fees=Decimal("84.5"),
            slippage=Decimal("21.0"),
            net_pnl=Decimal("394.5") if i % 2 == 0 else Decimal("-405.5"),
        )
        session.add(trade)
        trades.append(trade)
    session.flush()
    return trades


@pytest.fixture
def sample_csv_content() -> str:
    """Generate sample CSV content matching Kraken format."""
    lines = ["timestamp,open,high,low,close,volume,trades"]
    base_time = 1381017600  # 2013-10-06

    for i in range(50):
        timestamp = base_time + (i * 86400)  # Add days
        open_price = 100 + i * 0.5
        high = open_price + 2
        low = open_price - 1
        close = open_price + 1
        volume = 10 + i
        trades = 5 + i

        lines.append(
            f"{timestamp},{open_price},{high},{low},{close},{volume},{trades}"
        )

    return "\n".join(lines)


@pytest.fixture
def sample_csv_file(tmp_path, sample_csv_content: str):
    """Create a sample CSV file for testing."""
    csv_file = tmp_path / "test_candles.csv"
    csv_file.write_text(sample_csv_content)
    return csv_file


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame with OHLCV data."""
    data = {
        "timestamp": pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC"),
        "open": [100.0 + i * 0.5 for i in range(50)],
        "high": [102.0 + i * 0.5 for i in range(50)],
        "low": [99.0 + i * 0.5 for i in range(50)],
        "close": [101.0 + i * 0.5 for i in range(50)],
        "volume": [1000.0 + i * 10 for i in range(50)],
        "trades": [50 + i for i in range(50)],
    }
    return pd.DataFrame(data)


# Utility fixtures
@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
