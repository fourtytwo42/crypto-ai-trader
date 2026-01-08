"""Pump.fun trade normalization and candle generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from decimal import Decimal
from typing import Callable

import numpy as np
import pandas as pd
from sqlalchemy import func, select, case
from sqlalchemy.orm import Session

from src.pumpfun.models import (
    PumpCandle1m,
    PumpFeature1m,
    PumpSolPrice,
    PumpToken,
    PumpTrade,
    PumpBase,
)
from src.pumpfun.sol_price import fetch_sol_usd_at
from src.pumpfun.feature_extractor import extract_features_minute


@dataclass(frozen=True)
class PumpfunFilterConfig:
    min_age_minutes: int = 60
    active_age_minutes: int = 240
    min_total_trades: int = 10
    min_recent_trades: int = 10
    recent_window_minutes: int = 30


def ensure_pumpfun_tables(session: Session) -> None:
    PumpBase.metadata.create_all(session.get_bind())


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


def _price_lookup_with_cache(session: Session) -> Callable[[int], float | None]:
    cache: dict[int, float] = {}
    last_call = {"ts": 0.0}

    def _lookup(ts: int) -> float | None:
        bucket = ts - (ts % 3600)
        if bucket in cache:
            return cache[bucket]
        cached = session.execute(
            select(PumpSolPrice).where(PumpSolPrice.hour_timestamp == bucket).limit(1)
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
        session.add(PumpSolPrice(hour_timestamp=bucket, price_usd=price))
        cache[bucket] = price
        last_call["ts"] = datetime.now(tz=timezone.utc).timestamp()
        return price

    return _lookup


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


def _fetch_candidate_tokens(session: Session, config: PumpfunFilterConfig) -> list[str]:
    now_ts = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    min_age_ts = now_ts - config.min_age_minutes * 60 * 1000
    recent_cutoff = now_ts - config.recent_window_minutes * 60 * 1000

    recent_case = case(
        (PumpTrade.timestamp >= recent_cutoff, 1),
        else_=0,
    )

    stmt = (
        select(
            PumpToken.id,
            PumpToken.created_timestamp,
            func.count(PumpTrade.id).label("total_trades"),
            func.sum(recent_case).label("recent_trades"),
        )
        .join(PumpTrade, PumpTrade.token_id == PumpToken.id)
        .where(PumpToken.created_timestamp <= min_age_ts)
        .group_by(PumpToken.id, PumpToken.created_timestamp)
    )

    results = session.execute(stmt).all()
    eligible: list[str] = []
    for token_id, created_ts, total_trades, recent_trades in results:
        if total_trades < config.min_total_trades:
            continue
        age_minutes = (now_ts - created_ts) / (60 * 1000)
        if age_minutes < config.active_age_minutes:
            if recent_trades < config.min_recent_trades:
                continue
        eligible.append(token_id)
    return eligible


def _load_trades_for_token(session: Session, token_id: str) -> pd.DataFrame:
    stmt = (
        select(
            PumpTrade.timestamp,
            PumpTrade.price_sol,
            PumpTrade.price_usd,
            PumpTrade.amount_sol,
            PumpTrade.amount_usd,
        )
        .where(PumpTrade.token_id == token_id)
        .order_by(PumpTrade.timestamp)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["timestamp", "price_sol", "price_usd", "amount_sol", "amount_usd"])


def _persist_candles(
    session: Session,
    token_id: str,
    candles: pd.DataFrame,
    replace_existing: bool,
) -> None:
    if replace_existing:
        session.query(PumpCandle1m).filter(PumpCandle1m.token_id == token_id).delete()
    records = []
    for row in candles.itertuples(index=False):
        records.append(
            PumpCandle1m(
                token_id=token_id,
                timestamp=row.timestamp.to_pydatetime(),
                open=Decimal(str(row.open)),
                high=Decimal(str(row.high)),
                low=Decimal(str(row.low)),
                close=Decimal(str(row.close)),
                volume_usd=Decimal(str(row.volume_usd)),
                volume_sol=Decimal(str(row.volume_sol)),
                trades=int(row.trades),
            )
        )
    session.bulk_save_objects(records)


def _persist_features(
    session: Session,
    token_id: str,
    features: pd.DataFrame,
    replace_existing: bool,
) -> None:
    if replace_existing:
        session.query(PumpFeature1m).filter(PumpFeature1m.token_id == token_id).delete()
    records = []
    for row in features.itertuples(index=False):
        records.append(
            PumpFeature1m(
                token_id=token_id,
                timestamp=row.timestamp.to_pydatetime(),
                return_=Decimal(str(row.return_)) if not pd.isna(row.return_) else None,
                range=Decimal(str(row.range)) if not pd.isna(row.range) else None,
                body=Decimal(str(row.body)) if not pd.isna(row.body) else None,
                dlog_volume=Decimal(str(row.dlog_volume)) if not pd.isna(row.dlog_volume) else None,
                ret_mean_15=Decimal(str(row.ret_mean_15)) if not pd.isna(row.ret_mean_15) else None,
                ret_std_15=Decimal(str(row.ret_std_15)) if not pd.isna(row.ret_std_15) else None,
                ret_mean_60=Decimal(str(row.ret_mean_60)) if not pd.isna(row.ret_mean_60) else None,
                ret_std_60=Decimal(str(row.ret_std_60)) if not pd.isna(row.ret_std_60) else None,
            )
        )
    session.bulk_save_objects(records)


def sync_pumpfun_candles(
    session: Session,
    config: PumpfunFilterConfig,
    replace_existing: bool = False,
    max_tokens: int | None = None,
    price_lookup_enabled: bool = True,
) -> dict[str, int]:
    ensure_pumpfun_tables(session)
    price_lookup = (
        _price_lookup_with_cache(session) if price_lookup_enabled else (lambda _: None)
    )
    tokens = _fetch_candidate_tokens(session, config)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    processed = 0
    candles_written = 0
    features_written = 0

    for token_id in tokens:
        trades_df = _load_trades_for_token(session, token_id)
        if trades_df.empty:
            continue
        trades_df = _normalize_trades(trades_df, price_lookup)
        candles = build_minute_candles(trades_df)
        if candles.empty:
            continue

        features_df = extract_features_minute(candles, drop_na=True)
        features_df = features_df.rename(columns={"return": "return_"})

        _persist_candles(session, token_id, candles, replace_existing)
        _persist_features(session, token_id, features_df, replace_existing)
        processed += 1
        candles_written += len(candles)
        features_written += len(features_df)

    return {
        "tokens": processed,
        "candles": candles_written,
        "features": features_written,
    }
