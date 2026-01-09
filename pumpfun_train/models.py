"""Pump.fun database models."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class PumpBase(DeclarativeBase):
    """Base class for pump.fun models."""

    pass


class PumpToken(PumpBase):
    """Existing pump.fun token metadata."""

    __tablename__ = "tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    mint_address: Mapped[str] = mapped_column(String, unique=True)
    symbol: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    created_timestamp: Mapped[int] = mapped_column(BigInteger)
    king_of_the_hill_timestamp: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)

    trades: Mapped[list["PumpTrade"]] = relationship("PumpTrade", back_populates="token")


class PumpTrade(PumpBase):
    """Existing pump.fun trade stream."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id: Mapped[str] = mapped_column(String, ForeignKey("tokens.id"))
    price_sol: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    price_usd: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    amount_sol: Mapped[Decimal] = mapped_column(Numeric)
    amount_usd: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    base_amount: Mapped[Decimal] = mapped_column(Numeric)
    timestamp: Mapped[int] = mapped_column(BigInteger)

    token: Mapped[PumpToken] = relationship("PumpToken", back_populates="trades")


class PumpCandle1m(PumpBase):
    """Derived 1-minute candles for pump.fun tokens."""

    __tablename__ = "pump_candles_1m"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    volume_usd: Mapped[Decimal] = mapped_column(Numeric(24, 8))
    volume_sol: Mapped[Decimal] = mapped_column(Numeric(24, 8))
    trades: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_pump_candles_token_ts", "token_id", "timestamp"),
    )


class PumpFeature1m(PumpBase):
    """Derived 1-minute features for pump.fun tokens."""

    __tablename__ = "pump_features_1m"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    return_: Mapped[Decimal | None] = mapped_column("return", Numeric(20, 8), nullable=True)
    range: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    body: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    dlog_volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_mean_15: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_std_15: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_mean_60: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    ret_std_60: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("idx_pump_features_token_ts", "token_id", "timestamp"),
    )


class PumpSolPrice(PumpBase):
    """Cached SOL/USD hourly prices for normalization."""

    __tablename__ = "pump_sol_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hour_timestamp: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    price_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
