from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from config import get_model_dir
from db import get_session
from models import Token, TokenPrice
from pipeline import _normalize_trades, _price_lookup_with_cache, build_minute_candles, load_trades_for_token
from recent_features import build_recent_feature_df
from classifier import predict_direction_confidence
from model_loader import load_model_artifacts
from forecast import forecast_next_horizon


FEATURE_COLS = [
    "range",
    "body",
    "dlog_volume",
    "ret_mean_5",
    "ret_std_5",
    "ret_mean_15",
    "ret_std_15",
    "ret_mean_60",
    "ret_std_60",
    "minutes_since_launch",
    "minutes_since_koth",
    "has_koth",
    "koth_reached",
    "is_completed",
    "log_trades",
    "log_close",
    "log_volume",
]


@dataclass
class MinutePrediction:
    minutes: int
    direction: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_pct: float
    direction_confidence: float | None


@dataclass
class PredictionResult:
    mint: str
    token_id: str
    model_dir: str
    horizon: int
    predictions: list[MinutePrediction]


@dataclass
class TokenSnapshot:
    mint: str
    token_id: str
    symbol: str | None
    name: str | None
    created_timestamp: int
    king_of_the_hill_timestamp: int | None
    completed: bool
    current_price: float | None
    market_cap_usd: float | None
    last_trade_timestamp: int | None


@dataclass
class CandleSeries:
    token_id: str
    candles: list[dict[str, object]]


def _load_token_data(session, mint: str) -> tuple[str, dict[str, object], np.ndarray]:
    token = session.query(Token).filter(Token.mint_address == mint).one_or_none()
    if token is None:
        raise ValueError("mint not found")
    token_id = token.id
    token_meta = {
        "created_timestamp": token.created_timestamp,
        "king_of_the_hill_timestamp": token.king_of_the_hill_timestamp,
        "completed": token.completed,
    }
    trades_df = load_trades_for_token(session, token_id)
    if trades_df.empty:
        raise ValueError("no trades found for mint")
    price_lookup = _price_lookup_with_cache(session)
    normalized = _normalize_trades(trades_df, price_lookup)
    return token_id, token_meta, normalized


def get_token_snapshot(mint: str) -> TokenSnapshot:
    session = get_session()
    try:
        token = session.query(Token).filter(Token.mint_address == mint).one_or_none()
        if token is None:
            raise ValueError("mint not found")
        token_id = token.id
        price_row = session.query(TokenPrice).filter(TokenPrice.token_id == token_id).one_or_none()
        current_price = float(price_row.price_usd) if price_row is not None else None
        market_cap = float(price_row.market_cap_usd) if price_row and price_row.market_cap_usd is not None else None
        last_trade_ts = int(price_row.last_trade_timestamp) if price_row and price_row.last_trade_timestamp else None
        if current_price is None:
            token_id, token_meta, normalized = _load_token_data(session, mint)
            candles = build_minute_candles(normalized)
            if not candles.empty:
                current_price = float(candles["close"].iloc[-1])
        return TokenSnapshot(
            mint=mint,
            token_id=token_id,
            symbol=token.symbol,
            name=token.name,
            created_timestamp=token.created_timestamp,
            king_of_the_hill_timestamp=token.king_of_the_hill_timestamp,
            completed=token.completed,
            current_price=current_price,
            market_cap_usd=market_cap,
            last_trade_timestamp=last_trade_ts,
        )
    finally:
        session.close()


def get_candles(mint: str, limit: int = 240) -> CandleSeries:
    session = get_session()
    try:
        token_id, _token_meta, normalized = _load_token_data(session, mint)
    finally:
        session.close()
    candles = build_minute_candles(normalized)
    if candles.empty:
        raise ValueError("no candles available")
    candles = candles.tail(limit)
    items = [
        {
            "timestamp": row.timestamp.isoformat(),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume_usd": float(row.volume_usd),
            "trades": int(row.trades),
        }
        for row in candles.itertuples(index=False)
    ]
    return CandleSeries(token_id=token_id, candles=items)


def predict_minutes(mint: str, minutes: int) -> PredictionResult:
    model_dir = get_model_dir()
    bundle = load_model_artifacts(model_dir)
    config_meta = bundle.metadata.get("config", {})
    model_horizon = int(config_meta.get("horizon", 10))
    context_length = int(config_meta.get("context_length", 336))

    session = get_session()
    try:
        token_id, token_meta, normalized = _load_token_data(session, mint)
    finally:
        session.close()

    candles = build_minute_candles(normalized)
    if candles.empty:
        raise ValueError("no candles available")

    features_df = build_recent_feature_df(candles, token_meta=token_meta)
    if features_df.empty:
        raise ValueError("no features available")

    history_df = features_df.tail(context_length).reset_index(drop=True)
    feature_cols = [col for col in FEATURE_COLS if col in history_df.columns and col != "return"]

    direction_confidence, classifier_direction = predict_direction_confidence(history_df)

    preds = forecast_next_horizon(
        bundle.model,
        history_df,
        target_col="return",
        feature_cols=feature_cols,
        horizon=model_horizon,
    )
    preds = np.asarray(preds, dtype=float)
    if minutes > model_horizon and preds.size:
        extension = float(np.mean(preds[-3:])) if preds.size >= 3 else float(preds[-1])
        extra = np.full(minutes - model_horizon, extension, dtype=float)
        preds = np.concatenate([preds, extra])
    current_price = float(candles["close"].iloc[-1])

    max_minutes = min(minutes, preds.size)
    results: list[MinutePrediction] = []
    for m in range(1, max_minutes + 1):
        pred_return = float(np.sum(preds[:m]))
        predicted_price = float(current_price * np.exp(pred_return))
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100 if current_price else 0.0
        direction = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "FLAT"
        results.append(
            MinutePrediction(
                minutes=m,
                direction=direction,
                current_price=current_price,
                predicted_price=predicted_price,
                price_change=price_change,
                price_change_pct=price_change_pct,
                direction_confidence=direction_confidence,
            )
        )

    return PredictionResult(
        mint=mint,
        token_id=token_id,
        model_dir=str(model_dir),
        horizon=max_minutes,
        predictions=results,
    )
