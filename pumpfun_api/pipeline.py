from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import select

from models import SolPrice, Trade
from sol_price import fetch_sol_usd_at


def _normalize_trades(
    trades_df: pd.DataFrame,
    price_lookup: Callable[[int], float | None],
) -> pd.DataFrame:
    trades_df = trades_df.copy()
    if "amount_usd" in trades_df.columns:
        missing_price = trades_df["price_usd"].isna() | (trades_df["price_usd"] <= 0)
        has_amount = trades_df["amount_usd"].notna() & (trades_df["amount_usd"] > 0)
        has_sol = trades_df["amount_sol"].notna() & (trades_df["amount_sol"] > 0)
        derive_mask = missing_price & has_amount & has_sol
        trades_df.loc[derive_mask, "price_usd"] = (
            trades_df.loc[derive_mask, "amount_usd"].astype(float).values
            / trades_df.loc[derive_mask, "amount_sol"].astype(float).values
        )

    if "amount_usd" in trades_df.columns:
        missing_amount = trades_df["amount_usd"].isna() | (trades_df["amount_usd"] <= 0)
        has_price = trades_df["price_usd"].notna() & (trades_df["price_usd"] > 0)
        has_sol = trades_df["amount_sol"].notna() & (trades_df["amount_sol"] > 0)
        derive_mask = missing_amount & has_price & has_sol
        trades_df.loc[derive_mask, "amount_usd"] = (
            trades_df.loc[derive_mask, "price_usd"].astype(float).values
            * trades_df.loc[derive_mask, "amount_sol"].astype(float).values
        )

    needs_price = trades_df["price_usd"].isna() | (trades_df["price_usd"] <= 0)
    if needs_price.any():
        prices = []
        for ts_ms in trades_df.loc[needs_price, "timestamp"].astype(int).tolist():
            prices.append(price_lookup(ts_ms // 1000))
        price_series = pd.Series(prices, index=trades_df.loc[needs_price].index, dtype="float")
        trades_df.loc[needs_price, "price_usd"] = (
            trades_df.loc[needs_price, "price_sol"].astype(float).values * price_series.values
        )

    needs_amount = trades_df["amount_usd"].isna() | (trades_df["amount_usd"] <= 0)
    if needs_amount.any():
        prices = []
        for ts_ms in trades_df.loc[needs_amount, "timestamp"].astype(int).tolist():
            prices.append(price_lookup(ts_ms // 1000))
        price_series = pd.Series(prices, index=trades_df.loc[needs_amount].index, dtype="float")
        trades_df.loc[needs_amount, "amount_usd"] = (
            trades_df.loc[needs_amount, "amount_sol"].astype(float).values * price_series.values
        )

    for col in ["price_usd", "price_sol", "amount_usd", "amount_sol"]:
        if col in trades_df.columns:
            trades_df[col] = trades_df[col].astype(float)

    trades_df = trades_df.dropna(subset=["price_usd", "amount_usd"]).reset_index(drop=True)
    return trades_df


def _price_lookup_with_cache(session) -> Callable[[int], float | None]:
    cache: dict[int, float] = {}
    last_call = {"ts": 0.0}

    def _lookup(ts: int) -> float | None:
        bucket = ts - (ts % 3600)
        if bucket in cache:
            return cache[bucket]
        cached = session.execute(
            select(SolPrice).where(SolPrice.hour_timestamp == bucket).limit(1)
        ).scalar_one_or_none()
        if cached:
            cache[bucket] = float(cached.price_usd)
            return cache[bucket]
        now = datetime.now(tz=timezone.utc).timestamp()
        elapsed = now - last_call["ts"]
        if elapsed < 0.3:
            time.sleep(0.3 - elapsed)
        try:
            price = fetch_sol_usd_at(ts)
        except Exception:
            return None
        session.add(SolPrice(hour_timestamp=bucket, price_usd=price))
        cache[bucket] = price
        last_call["ts"] = datetime.now(tz=timezone.utc).timestamp()
        return price

    return _lookup


def load_trades_for_token(session, token_id: str) -> pd.DataFrame:
    stmt = (
        select(
            Trade.timestamp,
            Trade.price_sol,
            Trade.price_usd,
            Trade.amount_sol,
            Trade.amount_usd,
        )
        .where(Trade.token_id == token_id)
        .order_by(Trade.timestamp)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["timestamp", "price_sol", "price_usd", "amount_sol", "amount_usd"])


def build_minute_candles(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df

    trades_df = trades_df.sort_values("timestamp").reset_index(drop=True)
    trades_df["dt"] = pd.to_datetime(trades_df["timestamp"], unit="ms", utc=True)
    trades_df = trades_df.set_index("dt")

    agg = trades_df.resample("1min").agg(
        open=("price_usd", "first"),
        high=("price_usd", "max"),
        low=("price_usd", "min"),
        close=("price_usd", "last"),
        volume_usd=("amount_usd", "sum"),
        volume_sol=("amount_sol", "sum"),
        trades=("price_usd", "count"),
    )

    agg = agg.dropna(subset=["open", "close"]).copy()
    if agg.empty:
        return agg

    for col in ["open", "high", "low", "close", "volume_usd", "volume_sol"]:
        agg[col] = agg[col].astype(float)

    last_trade_ts = agg.index.max()
    full_index = pd.date_range(start=agg.index.min(), end=last_trade_ts, freq="1min", tz="UTC")
    agg = agg.reindex(full_index)
    agg["close"] = agg["close"].ffill()
    agg["open"] = agg["open"].fillna(agg["close"])
    agg["high"] = agg["high"].fillna(agg["close"])
    agg["low"] = agg["low"].fillna(agg["close"])
    agg["volume_usd"] = agg["volume_usd"].fillna(0.0)
    agg["volume_sol"] = agg["volume_sol"].fillna(0.0)
    agg["trades"] = agg["trades"].fillna(0).astype(int)

    agg = agg.reset_index().rename(columns={"index": "timestamp"})
    return agg
