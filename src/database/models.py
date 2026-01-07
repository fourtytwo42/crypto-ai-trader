"""SQLAlchemy database models for Bitcoin Trading Model.

Defines all database tables: candles, features, models,
predictions, backtests, and backtest_trades.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Candle(Base):
    """Raw OHLCV candle data from Kraken.

    Stores daily Bitcoin price data with volume and trade count.
    """

    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="BTC-USDT")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    features: Mapped[list["Feature"]] = relationship(
        "Feature", back_populates="candle", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_candles_symbol_timestamp"),
        Index("idx_candles_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_candles_timestamp", "timestamp"),
        Index("idx_candles_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Candle(id={self.id}, timestamp={self.timestamp}, close={self.close})>"


class Feature(Base):
    """Engineered features computed from candles.

    Contains 8 scale-free features for model training:
    - return: log(C_t / C_{t-1})
    - range: log(H_t / L_t)
    - body: log(C_t / O_t)
    - dlog_volume: log(V_t) - log(V_{t-1})
    - ret_mean_7: 7-day rolling mean of returns
    - ret_std_7: 7-day rolling std of returns
    - ret_mean_30: 30-day rolling mean of returns
    - ret_std_30: 30-day rolling std of returns
    """

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    candle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("candles.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="BTC-USDT")
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Scale-free features
    return_: Mapped[Decimal | None] = mapped_column(
        "return", Numeric(20, 8), nullable=True
    )
    range: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    body: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    dlog_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)

    # Rolling statistics
    ret_mean_7: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_std_7: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_mean_30: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_std_30: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    candle: Mapped["Candle"] = relationship("Candle", back_populates="features")

    __table_args__ = (
        UniqueConstraint("symbol", "timestamp", name="uq_features_symbol_timestamp"),
        Index("idx_features_symbol_timestamp", "symbol", "timestamp"),
        Index("idx_features_timestamp", "timestamp"),
        Index("idx_features_candle_id", "candle_id"),
    )

    def __repr__(self) -> str:
        return f"<Feature(id={self.id}, timestamp={self.timestamp})>"


class Model(Base):
    """Model metadata and configurations.

    Stores information about trained models including
    configuration, file paths, and performance metrics.
    """

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    scaler_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Training periods
    train_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    train_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    val_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    val_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="model", cascade="all, delete-orphan"
    )
    forecast_predictions: Mapped[list["ForecastPrediction"]] = relationship(
        "ForecastPrediction", back_populates="model", cascade="all, delete-orphan"
    )
    backtests: Mapped[list["Backtest"]] = relationship(
        "Backtest", back_populates="model", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_models_name", "name"),
        Index("idx_models_type", "model_type"),
        Index("idx_models_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Model(id={self.id}, name={self.name}, type={self.model_type})>"


class Prediction(Base):
    """Predictions generated by models.

    Stores quantile predictions and generated signals.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    prediction_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Quantile predictions
    q10: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    q50: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    q90: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)

    signal: Mapped[str | None] = mapped_column(String(10), nullable=True)
    threshold: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint("model_id", "prediction_timestamp", name="uq_prediction_model_timestamp"),
        Index("idx_predictions_model_id", "model_id"),
        Index("idx_predictions_timestamp", "timestamp"),
        Index("idx_predictions_prediction_timestamp", "prediction_timestamp"),
        Index("idx_predictions_signal", "signal"),
    )

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, signal={self.signal}, q50={self.q50})>"


class ForecastPrediction(Base):
    """Forecast predictions for price targets."""

    __tablename__ = "forecast_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, default="BTC-USDT")
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=24)
    data_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    target_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    predicted_direction: Mapped[str | None] = mapped_column(String(10), nullable=True)
    actual_close: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    accuracy_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    model: Mapped["Model"] = relationship("Model", back_populates="forecast_predictions")

    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "symbol",
            "data_timestamp",
            "horizon_hours",
            name="uq_forecast_pred_cache",
        ),
        Index("idx_forecast_preds_model_id", "model_id"),
        Index("idx_forecast_preds_symbol", "symbol"),
        Index("idx_forecast_preds_target_ts", "target_timestamp"),
        Index("idx_forecast_preds_predicted_at", "predicted_at"),
    )

    def __repr__(self) -> str:
        return f"<ForecastPrediction(id={self.id}, symbol={self.symbol})>"


class TrainingJob(Base):
    """Training job tracking."""

    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("models.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    model: Mapped["Model"] = relationship("Model")

    __table_args__ = (
        Index("idx_training_jobs_status", "status"),
        Index("idx_training_jobs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<TrainingJob(id={self.id}, status={self.status})>"


class Backtest(Base):
    """Backtest run metadata.

    Stores configuration and overall metrics for backtests.
    """

    __tablename__ = "backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    train_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    test_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fees: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    slippage: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    model: Mapped["Model"] = relationship("Model", back_populates="backtests")
    trades: Mapped[list["BacktestTrade"]] = relationship(
        "BacktestTrade", back_populates="backtest", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_backtests_model_id", "model_id"),
        Index("idx_backtests_start_date", "start_date"),
        Index("idx_backtests_end_date", "end_date"),
    )

    def __repr__(self) -> str:
        return f"<Backtest(id={self.id}, name={self.name})>"


class BacktestTrade(Base):
    """Individual trades from backtests.

    Records entry/exit prices, P&L, and costs for each trade.
    """

    __tablename__ = "backtest_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("backtests.id", ondelete="CASCADE"), nullable=False
    )
    entry_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    signal: Mapped[str] = mapped_column(String(10), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    exit_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    fees: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    slippage: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    net_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    backtest: Mapped["Backtest"] = relationship("Backtest", back_populates="trades")

    __table_args__ = (
        Index("idx_backtest_trades_backtest_id", "backtest_id"),
        Index("idx_backtest_trades_entry_timestamp", "entry_timestamp"),
        Index("idx_backtest_trades_exit_timestamp", "exit_timestamp"),
    )

    def __repr__(self) -> str:
        return f"<BacktestTrade(id={self.id}, signal={self.signal}, net_pnl={self.net_pnl})>"
