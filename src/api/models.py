"""API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_id: int = Field(..., description="Model ID to use")
    threshold: float = Field(0.005, ge=0, le=0.1, description="Base signal threshold")
    use_latest: bool = Field(True, description="Use latest data")
    data: dict[str, Any] | None = Field(None, description="Custom feature data")

    # Enhanced signal generation options
    volatility_adaptive: bool = Field(
        True, description="Enable volatility-adaptive threshold scaling"
    )
    current_volatility: float | None = Field(
        None, ge=0, description="Current volatility for adaptive threshold"
    )
    max_uncertainty_spread: float = Field(
        0.03, ge=0.005, le=0.1, description="Max q90-q10 spread before going flat"
    )
    include_position_size: bool = Field(
        False, description="Include position size in response based on conviction"
    )


class QuantilePrediction(BaseModel):
    q10: float
    q50: float
    q90: float


class PredictionResponse(BaseModel):
    model_id: int
    model_name: str
    prediction: QuantilePrediction
    signal: str = Field(..., pattern="^(long|short|flat)$")
    threshold: float
    prediction_timestamp: datetime
    timestamp: datetime
    confidence: float = Field(..., ge=0, le=1)

    # Enhanced signal information
    effective_threshold: float | None = Field(
        None, description="Volatility-adjusted threshold actually used"
    )
    uncertainty_spread: float | None = Field(
        None, description="Prediction uncertainty (q90 - q10)"
    )
    position_size: float | None = Field(
        None, ge=0, le=1, description="Suggested position size based on conviction"
    )
    conviction: float | None = Field(
        None, ge=0, le=1, description="Signal conviction score (|q50| / spread)"
    )


class ModelInfo(BaseModel):
    id: int
    name: str
    model_type: str
    version: str
    created_at: datetime
    metrics: dict[str, Any] | None = None


class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    num_trades: int
    turnover: float


class BacktestInfo(BaseModel):
    id: int
    model_id: int
    name: str
    start_date: datetime
    end_date: datetime
    train_window: int | None
    test_window: int | None
    fees: float | None
    slippage: float | None
    metrics: dict[str, Any]
    created_at: datetime


class TrainingRequest(BaseModel):
    model_type: str = Field("nhits", description="Model type (patchtst or nhits)")
    context_length: int = Field(168, ge=24, le=720, description="Context length in hours")
    horizon_hours: int = Field(24, ge=24, le=168, description="Prediction horizon in hours")
    hidden_size: int = Field(512, ge=64, le=2048, description="Hidden size")
    num_layers: int = Field(6, ge=1, le=24, description="Number of layers")
    patch_length: int = Field(16, ge=4, le=64, description="Patch length for PatchTST")
    stride: int = Field(8, ge=1, le=32, description="Stride for PatchTST")
    learning_rate: float = Field(0.0001, ge=0.00001, le=0.01, description="Learning rate")
    batch_size: int = Field(32, ge=8, le=256, description="Batch size")
    epochs: int = Field(100, ge=1, le=500, description="Training epochs")
    loss_type: str = Field("mae", description="Loss type")
    device: str | None = Field(None, description="Training device override")
    symbols: list[str] | None = Field(None, description="Symbols to train on")
    multi_asset: bool = Field(True, description="Train a multi-asset model")
    name_prefix: str | None = Field(None, description="Optional name prefix")
    max_vram_gb: float | None = Field(None, ge=1, le=64, description="Max VRAM usage")


class TrainingJobResponse(BaseModel):
    id: int
    status: str
    model_name: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    model_id: int | None = None


class ForecastPredictRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTC-USDT)")
    hours: int = Field(24, ge=24, le=168, description="Prediction horizon in hours")
    model_id: int | None = Field(None, description="Model ID to use")
    model_name: str | None = Field(None, description="Model name to use")
    use_cache: bool = Field(True, description="Use cached prediction if fresh")


class ForecastPredictResponse(BaseModel):
    symbol: str
    hours: int
    model_id: int
    model_name: str
    data_timestamp: datetime
    target_timestamp: datetime
    predicted_at: datetime
    current_price: float
    predicted_price: float
    price_change: float
    price_change_pct: float
    direction: str
    cache_hit: bool
    actual_price: float | None = None
    accuracy_pct: float | None = None


class ForecastHistoryItem(BaseModel):
    symbol: str
    hours: int
    model_id: int
    model_name: str
    data_timestamp: datetime
    target_timestamp: datetime
    predicted_at: datetime
    predicted_price: float
    actual_price: float | None = None
    accuracy_pct: float | None = None
