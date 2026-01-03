"""Tests for CLI command helpers."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine

from src.cli.commands import backtest_command, load_data_command, predict_command, train_command
from src.database.models import Base
from src.training.config import TrainingConfig
from src.training.model_factory import SimpleQuantileModel
from src.training.trainer import TrainingResult, save_model_artifacts


def _write_csv(csv_path: Path, rows: int = 40) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    base_ts = 1381017600
    for i in range(rows):
        ts = base_ts + i * 86400
        open_p = 100 + i
        high = open_p + 10
        low = open_p - 10
        close = open_p + 5
        volume = 10 + i
        lines.append(f"{ts},{open_p},{high},{low},{close},{volume}")
    csv_path.write_text("\n".join(lines))


def test_commands_flow(tmp_path, monkeypatch, test_db_url_with_schema, db_connect_args):
    engine = create_engine(test_db_url_with_schema, connect_args=db_connect_args)
    Base.metadata.create_all(bind=engine)
    engine.dispose()

    monkeypatch.setenv("DATABASE_URL", test_db_url_with_schema)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr("src.database.connection._db_manager", None)
    from src.config import get_settings

    get_settings.cache_clear()

    csv_path = tmp_path / "sample.csv"
    _write_csv(csv_path)

    result = load_data_command(str(csv_path), replace_existing=False)
    assert result["candles"] > 0

    train_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "return": [0.001] * 20,
        }
    )
    monkeypatch.setattr("src.cli.commands.prepare_training_data", lambda *_: (train_df, None))

    def fake_train_model(*args, **kwargs):
        return TrainingResult(
            model=SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9]),
            metrics={"mae": 0.0},
            model_path=Path("model.pt"),
            metadata_path=Path("metadata.json"),
            scaler_path=None,
        )

    monkeypatch.setattr("src.cli.commands.train_model", fake_train_model)

    metrics = train_command()
    assert "mae" in metrics

    model_dir = tmp_path / "model_artifacts"
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    save_model_artifacts(model, TrainingConfig(), {"mae": 0.1}, model_dir)

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "return": [0.001, 0.002, -0.001],
            "close": [100.0, 101.0, 102.0],
        }
    )

    pred = predict_command(str(model_dir), data)
    assert pred["signal"] in {"long", "short", "flat"}

    backtest = backtest_command(str(model_dir), data)
    assert "total_return" in backtest
