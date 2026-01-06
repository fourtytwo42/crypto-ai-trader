"""Database CRUD operations.

Provides functions for creating, reading, updating, and deleting
database records for all models.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import (
    Backtest,
    BacktestTrade,
    Candle,
    Feature,
    Model,
    Prediction,
)

logger = structlog.get_logger(__name__)


# Candle Operations
def create_candle(
    session: Session,
    symbol: str,
    timestamp: datetime,
    open_price: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal,
    trades: int | None = None,
) -> Candle:
    """Create a new candle record.

    Args:
        session: Database session.
        timestamp: Candle timestamp.
        open_price: Opening price.
        high: High price.
        low: Low price.
        close: Closing price.
        volume: Trading volume.
        trades: Number of trades (optional).

    Returns:
        Created candle.
    """
    candle = Candle(
        symbol=symbol,
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        trades=trades,
    )
    session.add(candle)
    session.flush()
    logger.debug("Created candle", candle_id=candle.id, timestamp=timestamp)
    return candle


def get_candle_by_timestamp(
    session: Session,
    timestamp: datetime,
    symbol: str | None = None,
) -> Candle | None:
    """Get candle by timestamp.

    Args:
        session: Database session.
        timestamp: Candle timestamp.

    Returns:
        Candle if found, None otherwise.
    """
    stmt = select(Candle).where(Candle.timestamp == timestamp)
    if symbol is not None:
        stmt = stmt.where(Candle.symbol == symbol)
    return session.execute(stmt).scalar_one_or_none()


def get_candles_in_range(
    session: Session,
    start: datetime,
    end: datetime,
    symbol: str | None = None,
) -> list[Candle]:
    """Get candles in date range.

    Args:
        session: Database session.
        start: Start timestamp (inclusive).
        end: End timestamp (inclusive).

    Returns:
        List of candles ordered by timestamp.
    """
    stmt = select(Candle).where(Candle.timestamp >= start).where(Candle.timestamp <= end)
    if symbol is not None:
        stmt = stmt.where(Candle.symbol == symbol)
    stmt = stmt.order_by(Candle.timestamp)
    return list(session.execute(stmt).scalars().all())


def get_latest_candles(
    session: Session,
    limit: int = 100,
    symbol: str | None = None,
) -> list[Candle]:
    """Get latest candles.

    Args:
        session: Database session.
        limit: Maximum number of candles.

    Returns:
        List of candles ordered by timestamp descending.
    """
    stmt = select(Candle)
    if symbol is not None:
        stmt = stmt.where(Candle.symbol == symbol)
    stmt = stmt.order_by(Candle.timestamp.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


def get_all_candles(session: Session, symbol: str | None = None) -> list[Candle]:
    """Get all candles ordered by timestamp.

    Args:
        session: Database session.

    Returns:
        List of all candles.
    """
    stmt = select(Candle)
    if symbol is not None:
        stmt = stmt.where(Candle.symbol == symbol)
    stmt = stmt.order_by(Candle.timestamp)
    return list(session.execute(stmt).scalars().all())


def get_available_symbols(session: Session) -> list[str]:
    """List distinct symbols available in candles."""
    stmt = select(Candle.symbol).distinct().order_by(Candle.symbol)
    return [row[0] for row in session.execute(stmt).all()]


def count_candles(session: Session) -> int:
    """Count total number of candles.

    Args:
        session: Database session.

    Returns:
        Number of candles.
    """
    stmt = select(Candle)
    return len(list(session.execute(stmt).scalars().all()))


def delete_all_candles(session: Session) -> int:
    """Delete all candles (cascades to features).

    Args:
        session: Database session.

    Returns:
        Number of deleted candles.
    """
    candles = session.query(Candle).all()
    count = len(candles)
    for candle in candles:
        session.delete(candle)
    session.flush()
    logger.info("Deleted all candles", count=count)
    return count


# Feature Operations
def create_feature(
    session: Session,
    candle_id: int,
    symbol: str,
    timestamp: datetime,
    return_: Decimal | None = None,
    range_: Decimal | None = None,
    body: Decimal | None = None,
    dlog_volume: Decimal | None = None,
    ret_mean_7: Decimal | None = None,
    ret_std_7: Decimal | None = None,
    ret_mean_30: Decimal | None = None,
    ret_std_30: Decimal | None = None,
) -> Feature:
    """Create a new feature record.

    Args:
        session: Database session.
        candle_id: Associated candle ID.
        timestamp: Feature timestamp.
        return_: Log return.
        range_: Log range.
        body: Log body.
        dlog_volume: Volume change.
        ret_mean_7: 7-day rolling mean.
        ret_std_7: 7-day rolling std.
        ret_mean_30: 30-day rolling mean.
        ret_std_30: 30-day rolling std.

    Returns:
        Created feature.
    """
    feature = Feature(
        candle_id=candle_id,
        symbol=symbol,
        timestamp=timestamp,
        return_=return_,
        range=range_,
        body=body,
        dlog_volume=dlog_volume,
        ret_mean_7=ret_mean_7,
        ret_std_7=ret_std_7,
        ret_mean_30=ret_mean_30,
        ret_std_30=ret_std_30,
    )
    session.add(feature)
    session.flush()
    logger.debug("Created feature", feature_id=feature.id, timestamp=timestamp)
    return feature


def get_features_in_range(
    session: Session,
    start: datetime,
    end: datetime,
    symbol: str | None = None,
) -> list[Feature]:
    """Get features in date range.

    Args:
        session: Database session.
        start: Start timestamp (inclusive).
        end: End timestamp (inclusive).

    Returns:
        List of features ordered by timestamp.
    """
    stmt = select(Feature).where(Feature.timestamp >= start).where(Feature.timestamp <= end)
    if symbol is not None:
        stmt = stmt.where(Feature.symbol == symbol)
    stmt = stmt.order_by(Feature.timestamp)
    return list(session.execute(stmt).scalars().all())


def get_feature_by_timestamp(
    session: Session,
    timestamp: datetime,
    symbol: str | None = None,
) -> Feature | None:
    """Get feature by timestamp and optional symbol."""
    stmt = select(Feature).where(Feature.timestamp == timestamp)
    if symbol is not None:
        stmt = stmt.where(Feature.symbol == symbol)
    return session.execute(stmt).scalar_one_or_none()


def get_latest_features(
    session: Session,
    limit: int = 100,
    symbol: str | None = None,
) -> list[Feature]:
    """Get latest features.

    Args:
        session: Database session.
        limit: Maximum number of features.

    Returns:
        List of features ordered by timestamp descending.
    """
    stmt = select(Feature)
    if symbol is not None:
        stmt = stmt.where(Feature.symbol == symbol)
    stmt = stmt.order_by(Feature.timestamp.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


# Model Operations
def create_model(
    session: Session,
    name: str,
    model_type: str,
    version: str,
    file_path: str,
    config: dict[str, Any],
    scaler_path: str | None = None,
    train_start: datetime | None = None,
    train_end: datetime | None = None,
    val_start: datetime | None = None,
    val_end: datetime | None = None,
    test_start: datetime | None = None,
    test_end: datetime | None = None,
    metrics: dict[str, Any] | None = None,
) -> Model:
    """Create a new model record.

    Args:
        session: Database session.
        name: Model name (unique).
        model_type: Model type (patchtst, nhits).
        version: Model version.
        file_path: Path to model weights.
        config: Training configuration.
        scaler_path: Path to scaler (optional).
        train_start: Training start date.
        train_end: Training end date.
        val_start: Validation start date.
        val_end: Validation end date.
        test_start: Test start date.
        test_end: Test end date.
        metrics: Model metrics (optional).

    Returns:
        Created model.
    """
    model = Model(
        name=name,
        model_type=model_type,
        version=version,
        file_path=file_path,
        scaler_path=scaler_path,
        config=config,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        metrics=metrics,
    )
    session.add(model)
    session.flush()
    logger.info("Created model", model_id=model.id, name=name)
    return model


def get_model_by_id(session: Session, model_id: int) -> Model | None:
    """Get model by ID.

    Args:
        session: Database session.
        model_id: Model ID.

    Returns:
        Model if found, None otherwise.
    """
    stmt = select(Model).where(Model.id == model_id)
    return session.execute(stmt).scalar_one_or_none()


def get_model_by_name(session: Session, name: str) -> Model | None:
    """Get model by name.

    Args:
        session: Database session.
        name: Model name.

    Returns:
        Model if found, None otherwise.
    """
    stmt = select(Model).where(Model.name == name)
    return session.execute(stmt).scalar_one_or_none()


def get_all_models(session: Session) -> list[Model]:
    """Get all models.

    Args:
        session: Database session.

    Returns:
        List of all models ordered by created_at descending.
    """
    stmt = select(Model).order_by(Model.created_at.desc())
    return list(session.execute(stmt).scalars().all())


def update_model_metrics(
    session: Session, model_id: int, metrics: dict[str, Any]
) -> Model | None:
    """Update model metrics.

    Args:
        session: Database session.
        model_id: Model ID.
        metrics: New metrics.

    Returns:
        Updated model or None if not found.
    """
    model = get_model_by_id(session, model_id)
    if model:
        model.metrics = metrics
        session.flush()
        logger.info("Updated model metrics", model_id=model_id)
    return model


# Prediction Operations
def create_prediction(
    session: Session,
    model_id: int,
    timestamp: datetime,
    prediction_timestamp: datetime,
    q10: Decimal | None = None,
    q50: Decimal | None = None,
    q90: Decimal | None = None,
    signal: str | None = None,
    threshold: Decimal | None = None,
) -> Prediction:
    """Create a new prediction record.

    Args:
        session: Database session.
        model_id: Associated model ID.
        timestamp: When prediction was made.
        prediction_timestamp: What we're predicting.
        q10: 10th percentile.
        q50: Median.
        q90: 90th percentile.
        signal: Generated signal.
        threshold: Signal threshold.

    Returns:
        Created prediction.
    """
    prediction = Prediction(
        model_id=model_id,
        timestamp=timestamp,
        prediction_timestamp=prediction_timestamp,
        q10=q10,
        q50=q50,
        q90=q90,
        signal=signal,
        threshold=threshold,
    )
    session.add(prediction)
    session.flush()
    logger.debug("Created prediction", prediction_id=prediction.id, signal=signal)
    return prediction


def get_predictions_by_model(
    session: Session, model_id: int, limit: int = 100
) -> list[Prediction]:
    """Get predictions for a model.

    Args:
        session: Database session.
        model_id: Model ID.
        limit: Maximum number of predictions.

    Returns:
        List of predictions ordered by prediction_timestamp descending.
    """
    stmt = (
        select(Prediction)
        .where(Prediction.model_id == model_id)
        .order_by(Prediction.prediction_timestamp.desc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def get_latest_prediction(session: Session, model_id: int) -> Prediction | None:
    """Get latest prediction for a model.

    Args:
        session: Database session.
        model_id: Model ID.

    Returns:
        Latest prediction or None.
    """
    stmt = (
        select(Prediction)
        .where(Prediction.model_id == model_id)
        .order_by(Prediction.prediction_timestamp.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


# Backtest Operations
def create_backtest(
    session: Session,
    model_id: int,
    name: str,
    start_date: datetime,
    end_date: datetime,
    metrics: dict[str, Any],
    train_window: int | None = None,
    test_window: int | None = None,
    fees: Decimal | None = None,
    slippage: Decimal | None = None,
) -> Backtest:
    """Create a new backtest record.

    Args:
        session: Database session.
        model_id: Associated model ID.
        name: Backtest name.
        start_date: Start date.
        end_date: End date.
        metrics: Backtest metrics.
        train_window: Training window in days.
        test_window: Test window in days.
        fees: Fee rate.
        slippage: Slippage rate.

    Returns:
        Created backtest.
    """
    backtest = Backtest(
        model_id=model_id,
        name=name,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        train_window=train_window,
        test_window=test_window,
        fees=fees,
        slippage=slippage,
    )
    session.add(backtest)
    session.flush()
    logger.info("Created backtest", backtest_id=backtest.id, name=name)
    return backtest


def get_backtest_by_id(session: Session, backtest_id: int) -> Backtest | None:
    """Get backtest by ID.

    Args:
        session: Database session.
        backtest_id: Backtest ID.

    Returns:
        Backtest if found, None otherwise.
    """
    stmt = select(Backtest).where(Backtest.id == backtest_id)
    return session.execute(stmt).scalar_one_or_none()


def get_backtests_by_model(session: Session, model_id: int) -> list[Backtest]:
    """Get backtests for a model.

    Args:
        session: Database session.
        model_id: Model ID.

    Returns:
        List of backtests ordered by created_at descending.
    """
    stmt = (
        select(Backtest)
        .where(Backtest.model_id == model_id)
        .order_by(Backtest.created_at.desc())
    )
    return list(session.execute(stmt).scalars().all())


def get_all_backtests(session: Session, limit: int = 20) -> list[Backtest]:
    """Get all backtests.

    Args:
        session: Database session.
        limit: Maximum number of backtests.

    Returns:
        List of backtests ordered by created_at descending.
    """
    stmt = select(Backtest).order_by(Backtest.created_at.desc()).limit(limit)
    return list(session.execute(stmt).scalars().all())


# BacktestTrade Operations
def create_backtest_trade(
    session: Session,
    backtest_id: int,
    entry_timestamp: datetime,
    exit_timestamp: datetime,
    signal: str,
    entry_price: Decimal,
    exit_price: Decimal,
    pnl: Decimal,
    fees: Decimal,
    slippage: Decimal,
    net_pnl: Decimal,
) -> BacktestTrade:
    """Create a new backtest trade record.

    Args:
        session: Database session.
        backtest_id: Associated backtest ID.
        entry_timestamp: Entry timestamp.
        exit_timestamp: Exit timestamp.
        signal: Trade direction.
        entry_price: Entry price.
        exit_price: Exit price.
        pnl: Gross P&L.
        fees: Fees.
        slippage: Slippage.
        net_pnl: Net P&L.

    Returns:
        Created trade.
    """
    trade = BacktestTrade(
        backtest_id=backtest_id,
        entry_timestamp=entry_timestamp,
        exit_timestamp=exit_timestamp,
        signal=signal,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        fees=fees,
        slippage=slippage,
        net_pnl=net_pnl,
    )
    session.add(trade)
    session.flush()
    return trade


def get_trades_by_backtest(session: Session, backtest_id: int) -> list[BacktestTrade]:
    """Get trades for a backtest.

    Args:
        session: Database session.
        backtest_id: Backtest ID.

    Returns:
        List of trades ordered by entry_timestamp.
    """
    stmt = (
        select(BacktestTrade)
        .where(BacktestTrade.backtest_id == backtest_id)
        .order_by(BacktestTrade.entry_timestamp)
    )
    return list(session.execute(stmt).scalars().all())
