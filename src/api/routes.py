"""API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.models import (
    BacktestInfo,
    ForecastHistoryItem,
    ForecastPredictRequest,
    ForecastPredictResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    QuantilePrediction,
    TrainingJobResponse,
    TrainingRequest,
)
from src.config import get_settings
from src.data.feature_extractor import get_feature_columns
from src.data.pipeline import prepare_training_data, prepare_training_data_multi
from src.database.operations import (
    create_forecast_prediction,
    create_prediction,
    create_training_job,
    get_all_backtests,
    get_all_models,
    get_backtest_by_id,
    get_backtests_by_model,
    get_candle_by_timestamp,
    get_forecast_predictions,
    get_latest_features,
    get_latest_prediction,
    get_model_by_id,
    get_model_by_name,
    get_predictions_by_model,
    get_trades_by_backtest,
    get_training_job,
    list_training_jobs,
    update_forecast_prediction_actuals,
    get_forecast_prediction_by_key,
    update_forecast_prediction,
    update_training_job,
    create_model,
)
from src.database.connection import get_db_manager
from src.forecasting.predict import forecast_next_horizon, build_future_timestamps
from src.forecasting.recent_features import (
    build_recent_feature_df_from_candles,
    fetch_recent_hourly_candles,
)
from src.forecasting.walk_forward_forecast import FORECAST_FEATURES
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
from src.training.config import TrainingConfig
from src.training.trainer import train_model
from src.training.targets import add_return_24h_target

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


def _resolve_model(db: Session, model_id: int | None, model_name: str | None):
    if model_id is not None:
        model = get_model_by_id(db, model_id)
        if model:
            return model
        raise_api_error(404, "MODEL_NOT_FOUND", "Model not found")
    if model_name:
        model = get_model_by_name(db, model_name)
        if model:
            return model
        raise_api_error(404, "MODEL_NOT_FOUND", "Model not found")
    models = get_all_models(db)
    if not models:
        raise_api_error(404, "MODEL_NOT_FOUND", "No models available")
    return models[0]


def _compute_accuracy_pct(predicted: float, actual: float) -> float | None:
    if actual == 0:
        return None
    return max(0.0, 1.0 - abs(predicted - actual) / actual) * 100.0


def _generate_model_name(config: TrainingConfig, prefix: str | None) -> str:
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [
        prefix or "model",
        config.model_type,
        f"ctx{config.context_length}",
        f"h{config.horizon}",
        f"hs{config.hidden_size}",
        f"l{config.num_layers}",
        f"pl{config.patch_length}",
        f"st{config.stride}",
        f"lr{config.learning_rate:g}",
        timestamp,
    ]
    return "_".join(parts)


def _run_training_job(job_id: int, model_name: str, request: TrainingRequest) -> None:
    settings = get_settings()
    db = get_db_manager()
    now = datetime.now(tz=timezone.utc)
    with db.session() as session:
        update_training_job(session, job_id, status="running", started_at=now)

    try:
        device = request.device or settings.train_device
        config = TrainingConfig(
            model_type=request.model_type,
            context_length=request.context_length,
            horizon=1,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            patch_length=request.patch_length,
            stride=request.stride,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            epochs=request.epochs,
            device=device,
            freq="H",
            data_frequency="1hour",
            max_vram_gb=request.max_vram_gb,
            loss_type=request.loss_type,
        )
        if request.horizon_hours != 24:
            raise ValueError("training only supports 24h horizon for now")

        with db.session() as session:
            if request.multi_asset:
                features_df, _ = prepare_training_data_multi(
                    session,
                    symbols=request.symbols,
                    normalize=False,
                )
                unique_id_col = "symbol"
            else:
                symbol = (request.symbols or [settings.data_symbol])[0]
                features_df, _ = prepare_training_data(
                    session,
                    normalize=False,
                    symbol=symbol,
                )
                unique_id_col = None

        if features_df.empty:
            raise ValueError("no features available for training")

        features_df = features_df.dropna().reset_index(drop=True)
        features_df = add_return_24h_target(features_df)
        features_df = features_df.dropna().reset_index(drop=True)
        if features_df.empty:
            raise ValueError("no valid rows after return_24h target")

        val_size = max(int(len(features_df) * settings.train_validation_split), 30)
        train_split = features_df.iloc[:-val_size].reset_index(drop=True)
        val_split = features_df.iloc[-val_size:].reset_index(drop=True)
        if train_split.empty or val_split.empty:
            raise ValueError("insufficient data for train/val split")

        feature_cols = [col for col in FORECAST_FEATURES if col in train_split.columns]
        model_dir = Path(settings.model_dir) / model_name

        result = train_model(
            config,
            train_split,
            val_split,
            model_dir=model_dir,
            target_col="return_24h",
            force_simple=False,
            feature_cols=feature_cols,
            save_artifacts=True,
            keep_best=False,
            unique_id_col=unique_id_col,
        )

        train_start = pd.to_datetime(train_split["timestamp"].min(), utc=True).to_pydatetime()
        train_end = pd.to_datetime(train_split["timestamp"].max(), utc=True).to_pydatetime()
        val_start = pd.to_datetime(val_split["timestamp"].min(), utc=True).to_pydatetime()
        val_end = pd.to_datetime(val_split["timestamp"].max(), utc=True).to_pydatetime()

        with db.session() as session:
            model = create_model(
                session,
                name=model_name,
                model_type=config.model_type,
                version="1.0",
                file_path=str(result.model_path),
                scaler_path=None,
                config=config.to_dict(),
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                metrics=result.metrics,
            )
            update_training_job(
                session,
                job_id,
                status="completed",
                finished_at=datetime.now(tz=timezone.utc),
                model_id=model.id,
                metrics=result.metrics,
            )
    except Exception as exc:
        with db.session() as session:
            update_training_job(
                session,
                job_id,
                status="failed",
                finished_at=datetime.now(tz=timezone.utc),
                error=str(exc),
            )


@router.post("/trainings", response_model=TrainingJobResponse)
async def create_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingJobResponse:
    model_name = _generate_model_name(
        TrainingConfig(
            model_type=request.model_type,
            context_length=request.context_length,
            horizon=1,
            hidden_size=request.hidden_size,
            num_layers=request.num_layers,
            patch_length=request.patch_length,
            stride=request.stride,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            epochs=request.epochs,
            device=request.device or get_settings().train_device,
            freq="H",
            data_frequency="1hour",
            max_vram_gb=request.max_vram_gb,
            loss_type=request.loss_type,
        ),
        request.name_prefix,
    )
    job = create_training_job(db, model_name=model_name, config=request.model_dump())
    background_tasks.add_task(_run_training_job, job.id, job.model_name, request)
    return TrainingJobResponse(
        id=job.id,
        status=job.status,
        model_name=job.model_name,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        metrics=job.metrics,
        error=job.error,
        model_id=job.model_id,
    )


@router.get("/trainings/{job_id}", response_model=TrainingJobResponse)
async def training_status(job_id: int, db: Session = Depends(get_db)) -> TrainingJobResponse:
    job = get_training_job(db, job_id)
    if not job:
        raise_api_error(404, "TRAINING_JOB_NOT_FOUND", "Training job not found")
    return TrainingJobResponse(
        id=job.id,
        status=job.status,
        model_name=job.model_name,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        metrics=job.metrics,
        error=job.error,
        model_id=job.model_id,
    )


@router.get("/trainings", response_model=list[TrainingJobResponse])
async def list_trainings(db: Session = Depends(get_db)) -> list[TrainingJobResponse]:
    jobs = list_training_jobs(db)
    return [
        TrainingJobResponse(
            id=job.id,
            status=job.status,
            model_name=job.model_name,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            metrics=job.metrics,
            error=job.error,
            model_id=job.model_id,
        )
        for job in jobs
    ]


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
        settings = get_settings()
        features = get_latest_features(db, limit=1, symbol=settings.data_symbol)
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


@router.post("/forecast/predict", response_model=ForecastPredictResponse)
async def forecast_predict(
    request: ForecastPredictRequest, db: Session = Depends(get_db)
) -> ForecastPredictResponse:
    if request.hours != 24:
        raise_api_error(400, "INVALID_HORIZON", "hours must be 24 for this model")

    model = _resolve_model(db, request.model_id, request.model_name)
    bundle = load_model_artifacts(Path(model.file_path).parent)
    config_meta = bundle.metadata.get("config", {})
    context_length = int(config_meta.get("context_length", 168))

    normalize_cols = get_feature_columns()
    if "return" in normalize_cols:
        normalize_cols = [col for col in normalize_cols if col != "return"]
    normalize_cols.append("log_close")
    normalize_cols = list(dict.fromkeys(normalize_cols))

    required_history = context_length + 24 + 30
    fetch_hours = required_history + 720
    candles = fetch_recent_hourly_candles(request.symbol, fetch_hours)
    if not candles:
        raise_api_error(400, "CANDLES_NOT_FOUND", "No candles available for symbol")

    symbol_df, normalizer = build_recent_feature_df_from_candles(
        candles, required_history, normalize_cols
    )
    if symbol_df.empty:
        raise_api_error(400, "FEATURES_NOT_FOUND", "No features available for symbol")
    symbol_df = symbol_df.dropna().reset_index(drop=True)
    symbol_df = add_return_24h_target(symbol_df)
    symbol_df = symbol_df.dropna().reset_index(drop=True)

    if len(symbol_df) < context_length:
        raise_api_error(
            400,
            "INSUFFICIENT_HISTORY",
            "Not enough history for prediction",
            {"available": len(symbol_df), "required": context_length},
        )

    history_df = symbol_df.tail(context_length).reset_index(drop=True)
    last_timestamp = pd.to_datetime(history_df["timestamp"].iloc[-1], utc=True).to_pydatetime()
    current_price = float(candles[-1].close)
    log_close_std = 1.0
    if normalizer and "log_close" in normalizer._feature_stats:
        log_close_std = normalizer._feature_stats["log_close"].get("std", 1.0)

    predicted_at = datetime.now(tz=timezone.utc)
    target_timestamp = build_future_timestamps(last_timestamp, request.hours)[-1]
    cache_hit = False
    cached = None
    if True:
        feature_cols = [col for col in FORECAST_FEATURES if col in history_df.columns]
        preds = forecast_next_horizon(
            bundle.model,
            history_df,
            target_col="return_24h",
            feature_cols=feature_cols,
            horizon=1,
        )
        pred_return = float(preds[-1])
        actual_log_return = pred_return * log_close_std
        predicted_close = float(current_price * np.exp(actual_log_return))
        target_timestamp = build_future_timestamps(last_timestamp, request.hours)[-1]

        predicted_direction = (
            "UP" if predicted_close > current_price else "DOWN" if predicted_close < current_price else "FLAT"
        )
        existing = get_forecast_prediction_by_key(
            db,
            model_id=model.id,
            symbol=request.symbol,
            horizon_hours=request.hours,
            data_timestamp=last_timestamp,
        )
        if existing:
            record = update_forecast_prediction(
                db,
                prediction=existing,
                predicted_at=predicted_at,
                predicted_close=Decimal(str(predicted_close)),
                predicted_direction=predicted_direction,
                target_timestamp=target_timestamp,
            )
        else:
            record = create_forecast_prediction(
                db,
                model_id=model.id,
                symbol=request.symbol,
                horizon_hours=request.hours,
                data_timestamp=last_timestamp,
                target_timestamp=target_timestamp,
                predicted_at=predicted_at,
                predicted_close=Decimal(str(predicted_close)),
                predicted_direction=predicted_direction,
            )

    actual_price = None
    accuracy_pct = None
    actual = get_candle_by_timestamp(db, target_timestamp, symbol=request.symbol)
    if actual:
        actual_price = float(actual.close)
        accuracy_pct = _compute_accuracy_pct(predicted_close, actual_price)
        if cached and cached.actual_close is None and accuracy_pct is not None:
            update_forecast_prediction_actuals(
                db,
                cached.id,
                actual_close=Decimal(str(actual_price)),
                accuracy_pct=Decimal(str(accuracy_pct)),
            )
        if not cached and accuracy_pct is not None:
            update_forecast_prediction_actuals(
                db,
                record.id,
                actual_close=Decimal(str(actual_price)),
                accuracy_pct=Decimal(str(accuracy_pct)),
            )

    price_change = predicted_close - current_price
    price_change_pct = (price_change / current_price * 100.0) if current_price else 0.0
    direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"

    return ForecastPredictResponse(
        symbol=request.symbol,
        hours=request.hours,
        model_id=model.id,
        model_name=model.name,
        data_timestamp=last_timestamp,
        target_timestamp=target_timestamp,
        predicted_at=predicted_at,
        current_price=current_price,
        predicted_price=predicted_close,
        price_change=price_change,
        price_change_pct=price_change_pct,
        direction=direction,
        cache_hit=cache_hit,
        actual_price=actual_price,
        accuracy_pct=accuracy_pct,
    )


@router.get("/forecast/history/{symbol}", response_model=list[ForecastHistoryItem])
async def forecast_history(
    symbol: str,
    model_id: int | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> list[ForecastHistoryItem]:
    predictions = get_forecast_predictions(db, symbol=symbol, model_id=model_id, limit=limit)
    items: list[ForecastHistoryItem] = []
    for pred in predictions:
        model = pred.model
        actual_price = float(pred.actual_close) if pred.actual_close is not None else None
        accuracy_pct = float(pred.accuracy_pct) if pred.accuracy_pct is not None else None
        items.append(
            ForecastHistoryItem(
                symbol=pred.symbol,
                hours=pred.horizon_hours,
                model_id=pred.model_id,
                model_name=model.name if model else "unknown",
                data_timestamp=pred.data_timestamp,
                target_timestamp=pred.target_timestamp,
                predicted_at=pred.predicted_at,
                predicted_price=float(pred.predicted_close),
                actual_price=actual_price,
                accuracy_pct=accuracy_pct,
            )
        )
    return items


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
