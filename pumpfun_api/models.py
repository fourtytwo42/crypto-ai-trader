from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, DateTime, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Token(Base):
    __tablename__ = "tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    mint_address: Mapped[str] = mapped_column(String)
    symbol: Mapped[str | None] = mapped_column(String, nullable=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_timestamp: Mapped[int] = mapped_column(BigInteger)
    king_of_the_hill_timestamp: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id: Mapped[str] = mapped_column(String)
    price_sol: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    price_usd: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    amount_sol: Mapped[Decimal] = mapped_column(Numeric)
    amount_usd: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    timestamp: Mapped[int] = mapped_column(BigInteger)


class SolPrice(Base):
    __tablename__ = "pump_sol_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hour_timestamp: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    price_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TokenPrice(Base):
    __tablename__ = "token_prices"

    token_id: Mapped[str] = mapped_column(String, primary_key=True)
    price_usd: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    market_cap_usd: Mapped[Decimal | None] = mapped_column(Numeric(24, 8), nullable=True)
    last_trade_timestamp: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
