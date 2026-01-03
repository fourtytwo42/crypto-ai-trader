"""API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.models import (
    BacktestInfo,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    QuantilePrediction,
)
from src.database.operations import (
    create_prediction,
    get_all_backtests,
    get_all_models,
    get_backtest_by_id,
    get_backtests_by_model,
    get_latest_features,
    get_latest_prediction,
    get_model_by_id,
    get_predictions_by_model,
    get_trades_by_backtest,
)
from src.prediction.model_loader import load_model_artifacts
from src.prediction.predictor import generate_predictions
from src.prediction.signal_generator import (
    SignalConfig,
    calculate_conviction,
    calculate_effective_threshold,
    calculate_uncertainty_spread,
    generate_signal,
    generate_signal_with_position_size,
)

router = APIRouter()


def raise_api_error(status_code: int, code: str, message: str, details: dict | None = None) -> None:
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message, "details": details or {}},
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models", response_model=list[ModelInfo])
async def list_models(db: Session = Depends(get_db)) -> list[ModelInfo]:
    models = get_all_models(db)
    return [
        ModelInfo(
            id=m.id,
            name=m.name,
            model_type=m.model_type,
            version=m.version,
            created_at=m.created_at,
            metrics=m.metrics,
        )
        for m in models
    ]


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: int, db: Session = Depends(get_db)) -> ModelInfo:
    model = get_model_by_id(db, model_id)
    if not model:
        raise_api_error(404, "MODEL_NOT_FOUND", "Model not found")
    return ModelInfo(
        id=model.id,
        name=model.name,
        model_type=model.model_type,
        version=model.version,
        created_at=model.created_at,
        metrics=model.metrics,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    model = get_model_by_id(db, request.model_id)
    if not model:
        raise_api_error(404, "MODEL_NOT_FOUND", "Model not found")

    model_dir = Path(model.file_path).parent
    bundle = load_model_artifacts(model_dir)

    if request.data:
        data = pd.DataFrame([request.data])
    elif request.use_latest:
        features = get_latest_features(db, limit=1)
        if not features:
            raise_api_error(400, "FEATURES_NOT_FOUND", "No features available")
        feature = features[0]
        data = pd.DataFrame(
            [
                {
                    "timestamp": feature.timestamp,
                    "return": float(feature.return_ or 0.0),
                }
            ]
        )

    else:
        raise_api_error(400, "DATA_REQUIRED", "No input data provided")

    predictions = generate_predictions(bundle.model, data)
    latest = predictions[-1]

    # Build enhanced signal config from request
    signal_config = SignalConfig(
        base_threshold=request.threshold,
        volatility_adaptive=request.volatility_adaptive,
        baseline_volatility=0.02,  # Default baseline
        uncertainty_enabled=True,
        max_uncertainty_spread=request.max_uncertainty_spread,
    )

    prediction_dict = {"q10": latest["q10"], "q50": latest["q50"], "q90": latest["q90"]}

    # Generate signal (with optional position sizing)
    position_size: float | None = None
    if request.include_position_size:
        signal, position_size = generate_signal_with_position_size(
            prediction_dict,
            threshold=request.threshold,
            current_volatility=request.current_volatility,
            config=signal_config,
        )
    else:
        signal = generate_signal(
            prediction_dict,
            threshold=request.threshold,
            current_volatility=request.current_volatility,
            config=signal_config,
        )

    # Calculate enhanced metrics for response
    uncertainty_spread = calculate_uncertainty_spread(prediction_dict)
    conviction = calculate_conviction(prediction_dict)
    effective_threshold: float | None = None
    if request.volatility_adaptive and request.current_volatility is not None:
        effective_threshold = calculate_effective_threshold(
            request.threshold,
            request.current_volatility,
            0.02,  # baseline
        )

    prediction_timestamp = datetime.now(tz=timezone.utc)
    create_prediction(
        db,
        model_id=model.id,
        timestamp=latest["timestamp"],
        prediction_timestamp=prediction_timestamp,
        q10=Decimal(str(latest["q10"])),
        q50=Decimal(str(latest["q50"])),
        q90=Decimal(str(latest["q90"])),
        signal=signal,
        threshold=Decimal(str(request.threshold)),
    )

    confidence = min(1.0, max(0.0, abs(latest["q50"]) * 100))
    return PredictionResponse(
        model_id=model.id,
        model_name=model.name,
        prediction=QuantilePrediction(q10=latest["q10"], q50=latest["q50"], q90=latest["q90"]),
        signal=signal,
        threshold=request.threshold,
        prediction_timestamp=prediction_timestamp,
        timestamp=latest["timestamp"],
        confidence=confidence,
        effective_threshold=effective_threshold,
        uncertainty_spread=uncertainty_spread,
        position_size=position_size,
        conviction=conviction,
    )


@router.get("/predictions/latest", response_model=list[PredictionResponse])
async def latest_predictions(db: Session = Depends(get_db)) -> list[PredictionResponse]:
    responses: list[PredictionResponse] = []
    for model in get_all_models(db):
        prediction = get_latest_prediction(db, model.id)
        if not prediction:
            continue
        responses.append(
            PredictionResponse(
                model_id=model.id,
                model_name=model.name,
                prediction=QuantilePrediction(
                    q10=float(prediction.q10 or 0.0),
                    q50=float(prediction.q50 or 0.0),
                    q90=float(prediction.q90 or 0.0),
                ),
                signal=prediction.signal or "flat",
                threshold=float(prediction.threshold or 0.0),
                prediction_timestamp=prediction.prediction_timestamp,
                timestamp=prediction.timestamp,
                confidence=min(1.0, abs(float(prediction.q50 or 0.0)) * 100),
            )
        )
    return responses


@router.get("/predictions/{model_id}", response_model=list[PredictionResponse])
async def predictions_by_model(model_id: int, db: Session = Depends(get_db)) -> list[PredictionResponse]:
    model = get_model_by_id(db, model_id)
    if not model:
        raise_api_error(404, "MODEL_NOT_FOUND", "Model not found")
    predictions = get_predictions_by_model(db, model_id)
    return [
        PredictionResponse(
            model_id=model.id,
            model_name=model.name,
            prediction=QuantilePrediction(
                q10=float(p.q10 or 0.0),
                q50=float(p.q50 or 0.0),
                q90=float(p.q90 or 0.0),
            ),
            signal=p.signal or "flat",
            threshold=float(p.threshold or 0.0),
            prediction_timestamp=p.prediction_timestamp,
            timestamp=p.timestamp,
            confidence=min(1.0, abs(float(p.q50 or 0.0)) * 100),
        )
        for p in predictions
    ]


@router.get("/backtests", response_model=list[BacktestInfo])
async def list_backtests(model_id: int | None = None, db: Session = Depends(get_db)) -> list[BacktestInfo]:
    backtests = (
        get_backtests_by_model(db, model_id) if model_id is not None else get_all_backtests(db)
    )
    return [
        BacktestInfo(
            id=b.id,
            model_id=b.model_id,
            name=b.name,
            start_date=b.start_date,
            end_date=b.end_date,
            train_window=b.train_window,
            test_window=b.test_window,
            fees=float(b.fees) if b.fees is not None else None,
            slippage=float(b.slippage) if b.slippage is not None else None,
            metrics=b.metrics,
            created_at=b.created_at,
        )
        for b in backtests
    ]


@router.get("/backtests/{backtest_id}")
async def backtest_details(backtest_id: int, db: Session = Depends(get_db)) -> dict[str, object]:
    backtest = get_backtest_by_id(db, backtest_id)
    if not backtest:
        raise_api_error(404, "BACKTEST_NOT_FOUND", "Backtest not found")
    trades = get_trades_by_backtest(db, backtest_id)
    return {
        "id": backtest.id,
        "model_id": backtest.model_id,
        "name": backtest.name,
        "start_date": backtest.start_date,
        "end_date": backtest.end_date,
        "train_window": backtest.train_window,
        "test_window": backtest.test_window,
        "fees": float(backtest.fees) if backtest.fees is not None else None,
        "slippage": float(backtest.slippage) if backtest.slippage is not None else None,
        "metrics": backtest.metrics,
        "trades": [
            {
                "id": t.id,
                "entry_timestamp": t.entry_timestamp,
                "exit_timestamp": t.exit_timestamp,
                "signal": t.signal,
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "pnl": float(t.pnl),
                "fees": float(t.fees),
                "slippage": float(t.slippage),
                "net_pnl": float(t.net_pnl),
            }
            for t in trades
        ],
        "created_at": backtest.created_at,
    }
