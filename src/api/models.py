"""API request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    model_id: int = Field(..., description="Model ID to use")
    threshold: float = Field(0.005, ge=0, le=0.1)
    use_latest: bool = Field(True, description="Use latest data")
    data: dict[str, Any] | None = Field(None, description="Custom feature data")


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
