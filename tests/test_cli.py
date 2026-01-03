"""CLI tests."""

from __future__ import annotations

from click.testing import CliRunner
from sqlalchemy import create_engine

from src.cli.cli import cli
from src.database.models import Base, Candle, Feature


def _seed_features(database_url: str, connect_args: dict[str, str]) -> None:
    from datetime import datetime, timezone
    from decimal import Decimal
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(database_url, connect_args=connect_args)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with SessionLocal() as session:
        candle = Candle(
            timestamp=datetime.now(tz=timezone.utc),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("10"),
            trades=None,
        )
        session.add(candle)
        session.flush()
        feature = Feature(
            candle_id=candle.id,
            timestamp=candle.timestamp,
            return_=Decimal("0.001"),
        )
        session.add(feature)
        session.commit()
    engine.dispose()


def test_cli_load_data_and_train(tmp_path, monkeypatch, test_db_url_with_schema, db_connect_args):
    engine = create_engine(test_db_url_with_schema, connect_args=db_connect_args)
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    monkeypatch.setenv("DATABASE_URL", test_db_url_with_schema)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr("src.database.connection._db_manager", None)
    from src.config import get_settings

    get_settings.cache_clear()
    csv_path = tmp_path / "sample.csv"

    rows = ["timestamp,open,high,low,close,volume"]
    base_ts = 1381017600
    for i in range(40):
        ts = base_ts + i * 86400
        open_p = 100 + i
        high = open_p + 10
        low = open_p - 10
        close = open_p + 5
        volume = 10 + i
        rows.append(f"{ts},{open_p},{high},{low},{close},{volume}")
    csv_path.write_text("\n".join(rows))

    runner = CliRunner()
    result = runner.invoke(cli, ["load-data", str(csv_path)])
    assert result.exit_code == 0


def test_cli_train_command(monkeypatch):
    monkeypatch.setattr("src.cli.cli.train_command", lambda: {"mae": 0.1})
    runner = CliRunner()
    result = runner.invoke(cli, ["train"])
    assert result.exit_code == 0


def test_cli_predict_and_backtest(tmp_path, test_db_url_with_schema, db_connect_args):
    from src.training.config import TrainingConfig
    from src.training.model_factory import SimpleQuantileModel
    from src.training.trainer import save_model_artifacts

    model_dir = tmp_path / "artifacts"
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    save_model_artifacts(model, TrainingConfig(), {"mae": 0.1}, model_dir)

    engine = create_engine(test_db_url_with_schema, connect_args=db_connect_args)
    Base.metadata.create_all(bind=engine)
    engine.dispose()
    _seed_features(test_db_url_with_schema, db_connect_args)

    import os
    from src.config import get_settings

    os.environ["DATABASE_URL"] = test_db_url_with_schema
    get_settings.cache_clear()
    import src.database.connection as connection_module

    connection_module._db_manager = None

    runner = CliRunner()
    pred_result = runner.invoke(cli, ["predict", "--model-dir", str(model_dir)])
    assert pred_result.exit_code == 0

    backtest_result = runner.invoke(cli, ["backtest", "--model-dir", str(model_dir)])
    assert backtest_result.exit_code == 0


def test_cli_menu_flag(monkeypatch):
    monkeypatch.setattr("src.cli.cli.menu_command", lambda: None)
    runner = CliRunner()
    result = runner.invoke(cli, ["--menu"])
    assert result.exit_code == 0
