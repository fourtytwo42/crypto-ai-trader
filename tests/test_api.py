"""API tests."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from src.api.dependencies import get_db
from src.api.main import app, http_exception_handler, unhandled_exception_handler
from src.database.models import Backtest, BacktestTrade, Base, Candle, Feature, Model
from src.training.config import TrainingConfig
from src.training.model_factory import SimpleQuantileModel
from src.training.trainer import save_model_artifacts


def _create_session(database_url: str, connect_args: dict[str, str]):
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return SessionLocal()


def test_api_endpoints(tmp_path, test_db_url_with_schema, db_connect_args):
    session = _create_session(test_db_url_with_schema, db_connect_args)

    model_dir = tmp_path / "models"
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    config = TrainingConfig()
    save_model_artifacts(model, config, {"mae": 0.1}, model_dir)

    db_model = Model(
        name="api_model",
        model_type="patchtst",
        version="1.0.0",
        file_path=str(model_dir / "model.pt"),
        scaler_path=None,
        config=config.to_dict(),
        metrics=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    session.add(db_model)
    session.flush()

    db_model_no_preds = Model(
        name="api_model_2",
        model_type="patchtst",
        version="1.0.0",
        file_path=str(model_dir / "model.pt"),
        scaler_path=None,
        config=config.to_dict(),
        metrics=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    session.add(db_model_no_preds)
    session.flush()

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

    backtest = Backtest(
        model_id=db_model.id,
        name="bt1",
        start_date=datetime(2022, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2022, 12, 31, tzinfo=timezone.utc),
        train_window=10,
        test_window=5,
        fees=Decimal("0.001"),
        slippage=Decimal("0.0005"),
        metrics={
            "total_return": 0.1,
            "sharpe_ratio": 1.0,
            "win_rate": 0.5,
            "avg_win": 0.02,
            "avg_loss": -0.01,
            "max_drawdown": -0.05,
            "num_trades": 2,
            "turnover": 2,
        },
    )
    session.add(backtest)
    session.flush()

    trade = BacktestTrade(
        backtest_id=backtest.id,
        entry_timestamp=datetime(2022, 1, 2, tzinfo=timezone.utc),
        exit_timestamp=datetime(2022, 1, 3, tzinfo=timezone.utc),
        signal="long",
        entry_price=Decimal("100"),
        exit_price=Decimal("110"),
        pnl=Decimal("10"),
        fees=Decimal("0.1"),
        slippage=Decimal("0.05"),
        net_pnl=Decimal("9.85"),
    )
    session.add(trade)
    session.commit()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200

    models = client.get("/models")
    assert models.status_code == 200
    assert len(models.json()) == 2

    model_detail = client.get(f"/models/{db_model.id}")
    assert model_detail.status_code == 200

    missing_model = client.get("/models/9999")
    assert missing_model.status_code == 404
    assert "error" in missing_model.json()

    prediction = client.post(
        "/predict",
        json={"model_id": db_model.id, "threshold": 0.005, "use_latest": True},
    )
    assert prediction.status_code == 200
    payload = prediction.json()
    assert payload["model_id"] == db_model.id

    latest_predictions = client.get("/predictions/latest")
    assert latest_predictions.status_code == 200
    assert len(latest_predictions.json()) >= 1

    model_predictions = client.get(f"/predictions/{db_model.id}")
    assert model_predictions.status_code == 200

    missing_predictions = client.get("/predictions/9999")
    assert missing_predictions.status_code == 404
    assert missing_predictions.json()["error"]["code"] == "MODEL_NOT_FOUND"

    missing_model_prediction = client.post(
        "/predict",
        json={"model_id": 9999, "threshold": 0.005, "use_latest": True},
    )
    assert missing_model_prediction.status_code == 404

    missing_data = client.post(
        "/predict",
        json={"model_id": db_model.id, "threshold": 0.005, "use_latest": False},
    )
    assert missing_data.status_code == 400
    assert missing_data.json()["error"]["code"] == "DATA_REQUIRED"

    backtests = client.get("/backtests")
    assert backtests.status_code == 200
    assert len(backtests.json()) == 1

    backtests_filtered = client.get(f"/backtests?model_id={db_model.id}")
    assert backtests_filtered.status_code == 200

    backtest_detail = client.get(f"/backtests/{backtest.id}")
    assert backtest_detail.status_code == 200

    missing_backtest = client.get("/backtests/9999")
    assert missing_backtest.status_code == 404
    assert missing_backtest.json()["error"]["code"] == "BACKTEST_NOT_FOUND"


@pytest.mark.anyio
async def test_api_error_handlers():
    scope = {"type": "http", "method": "GET", "path": "/"}
    request = Request(scope)

    response = await http_exception_handler(
        request,
        HTTPException(status_code=400, detail={"code": "BAD", "message": "bad", "details": {}}),
    )
    assert response.status_code == 400

    response = await unhandled_exception_handler(request, RuntimeError("boom"))
    assert response.status_code == 500


def test_api_dependency_get_db(monkeypatch):
    def fake_session():
        yield "session"

    monkeypatch.setattr("src.api.dependencies.get_session", fake_session)
    gen = get_db()
    assert next(gen) == "session"


def test_api_predict_no_features(tmp_path):
    db_path = tmp_path / "api_empty.db"
    session = _create_session(db_path)

    model_dir = tmp_path / "models"
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    config = TrainingConfig()
    save_model_artifacts(model, config, {"mae": 0.1}, model_dir)

    db_model = Model(
        name="api_model_empty",
        model_type="patchtst",
        version="1.0.0",
        file_path=str(model_dir / "model.pt"),
        scaler_path=None,
        config=config.to_dict(),
        metrics=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    session.add(db_model)
    session.commit()

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={"model_id": db_model.id, "threshold": 0.005, "use_latest": True},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "FEATURES_NOT_FOUND"
