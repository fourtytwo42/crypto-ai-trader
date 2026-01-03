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
