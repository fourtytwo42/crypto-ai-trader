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

try:
    import structlog
except ImportError:
    structlog = None

from pumpfun_train.models import (
    PumpCandle1m,
    PumpFeature1m,
    PumpSolPrice,
    PumpToken,
    PumpTrade,
    PumpBase,
)
from pumpfun_train.sol_price import fetch_sol_usd_at
from pumpfun_train.feature_extractor import extract_features_minute


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


def _fetch_candidate_tokens(session: Session, config: PumpfunFilterConfig, only_incremental: bool = True) -> list[str]:
    """Fetch candidate tokens for syncing.
    
    Args:
        session: Database session
        config: Filter configuration
        only_incremental: If True, only return tokens that have new trades since last sync
    """
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
            func.max(PumpTrade.timestamp).label("latest_trade_ts"),
        )
        .join(PumpTrade, PumpTrade.token_id == PumpToken.id)
        .where(PumpToken.created_timestamp <= min_age_ts)
        .group_by(PumpToken.id, PumpToken.created_timestamp)
    )

    results = session.execute(stmt).all()
    eligible: list[str] = []
    
    for token_id, created_ts, total_trades, recent_trades, latest_trade_ts in results:
        if total_trades < config.min_total_trades:
            continue
        age_minutes = (now_ts - created_ts) / (60 * 1000)
        if age_minutes < config.active_age_minutes:
            if recent_trades < config.min_recent_trades:
                continue
        
        # If incremental mode, check if there are new trades since last sync
        if only_incremental:
            latest_candle_ts = _get_latest_candle_timestamp(session, token_id)
            if latest_candle_ts is not None:
                # Only include if there are new trades (with 2 minute buffer)
                if latest_trade_ts <= (latest_candle_ts + 120000):  # +2 minutes in ms
                    continue  # Skip tokens that are up-to-date
        
        eligible.append(token_id)
    return eligible


def _get_latest_candle_timestamp(session: Session, token_id: str) -> int | None:
    """Get the latest candle timestamp for a token, or None if no candles exist."""
    stmt = (
        select(func.max(PumpCandle1m.timestamp))
        .where(PumpCandle1m.token_id == token_id)
    )
    result = session.execute(stmt).scalar_one_or_none()
    if result is None:
        return None
    # Convert datetime to milliseconds timestamp
    if isinstance(result, datetime):
        return int(result.timestamp() * 1000)
    return None


def _load_trades_for_token(session: Session, token_id: str, since_timestamp: int | None = None) -> pd.DataFrame:
    """Load trades for a token, optionally only those after since_timestamp."""
    stmt = (
        select(
            PumpTrade.timestamp,
            PumpTrade.price_sol,
            PumpTrade.price_usd,
            PumpTrade.amount_sol,
            PumpTrade.amount_usd,
        )
        .where(PumpTrade.token_id == token_id)
    )
    if since_timestamp is not None:
        # Only load trades after the latest candle (plus 1 minute buffer for overlap)
        stmt = stmt.where(PumpTrade.timestamp >= (since_timestamp - 60000))  # -1 minute in ms
    stmt = stmt.order_by(PumpTrade.timestamp)
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["timestamp", "price_sol", "price_usd", "amount_sol", "amount_usd"])


def _persist_candles(
    session: Session,
    token_id: str,
    candles: pd.DataFrame,
    replace_existing: bool,
) -> int:
    """Persist candles to database. Returns number of candles inserted."""
    if replace_existing:
        session.query(PumpCandle1m).filter(PumpCandle1m.token_id == token_id).delete()
        candles_to_insert = candles
    else:
        # Check for existing candles to avoid duplicates
        if candles.empty:
            return 0
        
        # Convert timestamps to datetime for comparison
        candle_timestamps = pd.to_datetime(candles['timestamp'], utc=True).dt.tz_localize(None) if candles['timestamp'].dtype != 'datetime64[ns, UTC]' else candles['timestamp'].dt.tz_localize(None)
        
        # Query existing timestamps for this token
        existing_stmt = (
            select(PumpCandle1m.timestamp)
            .where(PumpCandle1m.token_id == token_id)
            .where(
                PumpCandle1m.timestamp.in_(
                    [pd.Timestamp(ts).to_pydatetime() for ts in candle_timestamps.unique()]
                )
            )
        )
        existing_timestamps = {pd.Timestamp(ts).to_pydatetime() for ts in session.execute(existing_stmt).scalars().all()}
        
        # Filter out candles that already exist
        def is_new_candle(row):
            ts = pd.Timestamp(row.timestamp).to_pydatetime() if isinstance(row.timestamp, pd.Timestamp) else row.timestamp
            return ts not in existing_timestamps
        
        candles_to_insert = candles[candles.apply(is_new_candle, axis=1)]
    
    if candles_to_insert.empty:
        return 0
    
    records = []
    for row in candles_to_insert.itertuples(index=False):
        ts = row.timestamp.to_pydatetime() if isinstance(row.timestamp, pd.Timestamp) else row.timestamp
        records.append(
            PumpCandle1m(
                token_id=token_id,
                timestamp=ts,
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
    return len(records)


def _persist_features(
    session: Session,
    token_id: str,
    features: pd.DataFrame,
    replace_existing: bool,
) -> int:
    """Persist features to database. Returns number of features inserted."""
    if replace_existing:
        session.query(PumpFeature1m).filter(PumpFeature1m.token_id == token_id).delete()
        features_to_insert = features
    else:
        # Check for existing features to avoid duplicates
        if features.empty:
            return 0
        
        # Convert timestamps to datetime for comparison
        feature_timestamps = pd.to_datetime(features['timestamp'], utc=True).dt.tz_localize(None) if features['timestamp'].dtype != 'datetime64[ns, UTC]' else features['timestamp'].dt.tz_localize(None)
        
        # Query existing timestamps for this token
        existing_stmt = (
            select(PumpFeature1m.timestamp)
            .where(PumpFeature1m.token_id == token_id)
            .where(
                PumpFeature1m.timestamp.in_(
                    [pd.Timestamp(ts).to_pydatetime() for ts in feature_timestamps.unique()]
                )
            )
        )
        existing_timestamps = {pd.Timestamp(ts).to_pydatetime() for ts in session.execute(existing_stmt).scalars().all()}
        
        # Filter out features that already exist
        def is_new_feature(row):
            ts = pd.Timestamp(row.timestamp).to_pydatetime() if isinstance(row.timestamp, pd.Timestamp) else row.timestamp
            return ts not in existing_timestamps
        
        features_to_insert = features[features.apply(is_new_feature, axis=1)]
    
    if features_to_insert.empty:
        return 0
    
    records = []
    for row in features_to_insert.itertuples(index=False):
        ts = row.timestamp.to_pydatetime() if isinstance(row.timestamp, pd.Timestamp) else row.timestamp
        records.append(
            PumpFeature1m(
                token_id=token_id,
                timestamp=ts,
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
    return len(records)


def _process_single_token_worker(args: tuple) -> tuple[str, int, int, bool]:
    """Worker function for processing a single token (must be at module level for multiprocessing).
    
    Args:
        args: (token_id, database_url, config_dict, replace_existing, price_lookup_enabled)
    
    Returns:
        (token_id, candles_count, features_count, success)
    """
    token_id, database_url, config_dict, replace_existing, price_lookup_enabled = args
    
    # Reconstruct config from dict (dataclasses aren't pickleable)
    config = PumpfunFilterConfig(**config_dict)
    
    from pumpfun_train.db import DatabaseManager
    from pumpfun_train.sol_price import fetch_sol_usd_at
    
    db = DatabaseManager(database_url)
    with db.session() as session:
        try:
            # Create price lookup function for this worker
            if price_lookup_enabled:
                price_lookup_cache = {}
                last_call = {"ts": 0.0}
                
                def _price_lookup(ts: int) -> float | None:
                    from datetime import datetime, timezone
                    bucket = (ts // 3600) * 3600
                    if bucket in price_lookup_cache:
                        return price_lookup_cache[bucket]
                    cached = session.execute(
                        select(PumpSolPrice).where(PumpSolPrice.hour_timestamp == bucket).limit(1)
                    ).scalar_one_or_none()
                    if cached:
                        price_lookup_cache[bucket] = float(cached.price_usd)
                        return price_lookup_cache[bucket]
                    now = datetime.now(tz=timezone.utc).timestamp()
                    elapsed = now - last_call["ts"]
                    if elapsed < 0.3:
                        time.sleep(0.3 - elapsed)
                    try:
                        price = fetch_sol_usd_at(ts)
                    except Exception:
                        return None
                    session.add(PumpSolPrice(hour_timestamp=bucket, price_usd=price))
                    price_lookup_cache[bucket] = price
                    last_call["ts"] = datetime.now(tz=timezone.utc).timestamp()
                    return price
            else:
                _price_lookup = lambda _: None
            
            # Check if we already have candles for this token
            latest_candle_ts = _get_latest_candle_timestamp(session, token_id) if not replace_existing else None
            
            # Only load trades that haven't been processed yet
            trades_df = _load_trades_for_token(session, token_id, since_timestamp=latest_candle_ts)
            if trades_df.empty:
                return (token_id, 0, 0, False)  # No new trades to process
            
            trades_df = _normalize_trades(trades_df, _price_lookup)
            candles = build_minute_candles(trades_df)
            if candles.empty:
                return (token_id, 0, 0, False)

            # If we have existing candles, merge them to avoid gaps
            # But filter out candles that would duplicate existing ones
            if latest_candle_ts is not None and not candles.empty:
                latest_candle_dt = pd.to_datetime(latest_candle_ts, unit='ms', utc=True)
                # Only keep candles after the latest existing one (with 1 minute overlap for merge)
                candles = candles[candles['timestamp'] >= (latest_candle_dt - pd.Timedelta(minutes=1))]
            
            if candles.empty:
                return (token_id, 0, 0, False)

            features_df = extract_features_minute(candles, drop_na=True)
            features_df = features_df.rename(columns={"return": "return_"})

            # Use replace_existing flag to control behavior
            # The persist functions will check for duplicates if replace_existing=False
            candles_inserted = _persist_candles(session, token_id, candles, replace_existing)
            features_inserted = _persist_features(session, token_id, features_df, replace_existing)
            session.commit()
            
            return (token_id, candles_inserted, features_inserted, True)
        except Exception:
            session.rollback()
            # Return failure but don't crash
            return (token_id, 0, 0, False)


def sync_pumpfun_candles(
    session: Session,
    config: PumpfunFilterConfig,
    replace_existing: bool = False,
    max_tokens: int | None = None,
    price_lookup_enabled: bool = True,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Sync pump.fun trades into minute candles/features with parallel processing.
    
    Args:
        session: Database session
        config: Filter configuration
        replace_existing: Whether to replace existing data
        max_tokens: Maximum number of tokens to process
        price_lookup_enabled: Whether to enable SOL price lookup
        max_workers: Number of parallel workers (default: CPU count)
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm not available
        tqdm = lambda x, **kwargs: x
    
    ensure_pumpfun_tables(session)
    
    # Get database URL for worker processes
    from pumpfun_train.config_db import get_pumpfun_database_url
    database_url = get_pumpfun_database_url()
    
    # Only fetch tokens that need syncing (have new trades)
    tokens = _fetch_candidate_tokens(session, config, only_incremental=not replace_existing)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    
    if not tokens:
        return {"tokens": 0, "candles": 0, "features": 0}

    # Determine number of workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(tokens), 8)  # Cap at 8 to avoid DB overload
    
    processed = 0
    candles_written = 0
    features_written = 0

    # Convert config to dict for pickling (dataclasses aren't pickleable)
    config_dict = {
        "min_age_minutes": config.min_age_minutes,
        "active_age_minutes": config.active_age_minutes,
        "min_total_trades": config.min_total_trades,
        "min_recent_trades": config.min_recent_trades,
        "recent_window_minutes": config.recent_window_minutes,
    }

    # Progress bar for token processing
    pbar = tqdm(
        total=len(tokens),
        desc=f"Syncing tokens ({max_workers} workers)",
        unit="token",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Prepare arguments for workers
    worker_args = [
        (token_id, database_url, config_dict, replace_existing, price_lookup_enabled)
        for token_id in tokens
    ]
    
    # Process tokens in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_token = {
            executor.submit(_process_single_token_worker, args): args[0]
            for args in worker_args
        }
        
        # Process completed tasks
        for future in as_completed(future_to_token):
            try:
                token_id, candles_count, features_count, success = future.result()
                if success:
                    processed += 1
                    candles_written += candles_count
                    features_written += features_count
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    "processed": processed,
                    "candles": candles_written,
                    "features": features_written,
                    "workers": max_workers
                })
            except Exception as e:
                # Log error but continue processing
                pbar.update(1)
                if structlog:
                    logger = structlog.get_logger(__name__)
                    logger.warning("Token processing failed", token_id=future_to_token.get(future, "unknown"), error=str(e))
    
    pbar.close()

    return {
        "tokens": processed,
        "candles": candles_written,
        "features": features_written,
    }
