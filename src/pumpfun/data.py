"""Pump.fun training data preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.normalizer import RollingNormalizer
from src.pumpfun.models import PumpCandle1m, PumpFeature1m, PumpToken

PUMPFUN_FEATURE_COLUMNS = [
    "range",
    "body",
    "dlog_volume",
    "ret_mean_15",
    "ret_std_15",
    "ret_mean_60",
    "ret_std_60",
    "ret_mean_5",
    "ret_std_5",
    "minutes_since_launch",
    "minutes_since_koth",
    "has_koth",
    "koth_reached",
    "is_completed",
    "log_trades",
]

PUMPFUN_NORMALIZED_COLUMNS = [
    "range",
    "body",
    "dlog_volume",
    "ret_mean_15",
    "ret_std_15",
    "ret_mean_60",
    "ret_std_60",
    "ret_mean_5",
    "ret_std_5",
    "minutes_since_launch",
    "minutes_since_koth",
    "log_trades",
]


@dataclass
class PumpfunTrainingData:
    df: pd.DataFrame
    normalizers: dict[str, RollingNormalizer]


def _timestamp_to_utc(ts: int | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    unit = "ms" if ts > 1_000_000_000_000 else "s"
    return pd.to_datetime(ts, unit=unit, utc=True)


def _load_token_features(session: Session, token_id: str) -> pd.DataFrame:
    stmt = (
        select(
            PumpCandle1m.timestamp,
            PumpCandle1m.close,
            PumpCandle1m.volume_usd,
            PumpCandle1m.trades,
            PumpFeature1m.return_,
            PumpFeature1m.range,
            PumpFeature1m.body,
            PumpFeature1m.dlog_volume,
            PumpFeature1m.ret_mean_15,
            PumpFeature1m.ret_std_15,
            PumpFeature1m.ret_mean_60,
            PumpFeature1m.ret_std_60,
            PumpToken.created_timestamp,
            PumpToken.king_of_the_hill_timestamp,
            PumpToken.completed,
        )
        .join(
            PumpFeature1m,
            (PumpFeature1m.token_id == PumpCandle1m.token_id)
            & (PumpFeature1m.timestamp == PumpCandle1m.timestamp),
        )
        .join(PumpToken, PumpToken.id == PumpCandle1m.token_id)
        .where(PumpCandle1m.token_id == token_id)
        .order_by(PumpCandle1m.timestamp)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "close",
            "volume_usd",
            "trades",
            "return",
            "range",
            "body",
            "dlog_volume",
            "ret_mean_15",
            "ret_std_15",
            "ret_mean_60",
            "ret_std_60",
            "created_timestamp",
            "king_of_the_hill_timestamp",
            "completed",
        ],
    )
    for col in [
        "close",
        "volume_usd",
        "trades",
        "return",
        "range",
        "body",
        "dlog_volume",
        "ret_mean_15",
        "ret_std_15",
        "ret_mean_60",
        "ret_std_60",
    ]:
        df[col] = df[col].astype(float)
    df["ret_mean_5"] = df["return"].rolling(window=5).mean()
    df["ret_std_5"] = df["return"].rolling(window=5).std()
    created_ts = int(df["created_timestamp"].iloc[0])
    koth_ts = df["king_of_the_hill_timestamp"].iloc[0]
    completed = bool(df["completed"].iloc[0])
    created_dt = _timestamp_to_utc(created_ts)
    koth_dt = _timestamp_to_utc(int(koth_ts)) if pd.notna(koth_ts) else None
    df["minutes_since_launch"] = (
        (df["timestamp"] - created_dt).dt.total_seconds() / 60 if created_dt is not None else 0.0
    )
    if koth_dt is not None:
        df["minutes_since_koth"] = (df["timestamp"] - koth_dt).dt.total_seconds() / 60
        df["has_koth"] = 1.0
        df["koth_reached"] = (df["timestamp"] >= koth_dt).astype(float)
    else:
        df["minutes_since_koth"] = 0.0
        df["has_koth"] = 0.0
        df["koth_reached"] = 0.0
    df["is_completed"] = 1.0 if completed else 0.0
    return df


def prepare_pumpfun_training_data(
    session: Session,
    token_ids: Iterable[str],
    normalize: bool = True,
) -> PumpfunTrainingData:
    frames: list[pd.DataFrame] = []
    normalizers: dict[str, RollingNormalizer] = {}

    for token_id in token_ids:
        df = _load_token_features(session, token_id)
        if df.empty:
            continue
        df = df.copy()
        df["token_id"] = token_id
        eps = 1e-10
        df["log_close"] = np.log(df["close"].clip(lower=eps))
        df["log_volume"] = np.log(df["volume_usd"].clip(lower=eps))
        df["log_trades"] = np.log(df["trades"].clip(lower=1.0))
        df["raw_close"] = df["close"]
        df["raw_log_close"] = df["log_close"]
        if normalize:
            normalizer = RollingNormalizer(window=60)
            normalize_cols = [*PUMPFUN_NORMALIZED_COLUMNS, "log_close", "log_volume"]
            df = normalizer.fit_transform(df, normalize_cols)
            normalizers[token_id] = normalizer
        frames.append(df)

    if not frames:
        return PumpfunTrainingData(df=pd.DataFrame(), normalizers={})

    combined = pd.concat(frames, ignore_index=True)
    return PumpfunTrainingData(df=combined, normalizers=normalizers)


def add_return_target(df: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
    df["return_horizon"] = df.groupby("token_id")["log_close"].shift(-horizon_minutes) - df["log_close"]
    return df
