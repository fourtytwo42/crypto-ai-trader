from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
import os
import math
import gc
import numpy as np

from config import get_model_dir, get_regression_models_dir
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
    current_price: float | None
    predicted_price: float | None
    price_change: float | None
    price_change_pct: float | None
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
        features_df = build_recent_feature_df(candles, token_meta=token_meta, allow_sparse=True)
    if features_df.empty:
        raise ValueError("no features available")

    direction_confidence, classifier_direction = predict_direction_confidence(features_df)
    current_price = float(candles["close"].iloc[-1])
    if not math.isfinite(current_price):
        current_price = None

    max_minutes = max(1, minutes)
    horizon_cap = 20
    horizons_needed = sorted({min(m, horizon_cap) for m in range(1, max_minutes + 1)})
    horizon_predictions: dict[int, float] = {}

    def _load_bundle_for_horizon(horizon: int) -> tuple[Path, object, int, int, list[str], object]:
        reg_root = get_regression_models_dir()
        model_path = reg_root / f"h{horizon:02d}"
        if not model_path.exists():
            model_path = model_dir
        bundle = _get_cached_bundle(model_path)
        config_meta = bundle.metadata.get("config", {})
        model_horizon = int(config_meta.get("horizon", horizon))
        context_length = int(config_meta.get("context_length", 336))
        history_df = features_df.tail(context_length).reset_index(drop=True)
        feature_cols = [col for col in FEATURE_COLS if col in history_df.columns and col != "return"]
        return model_path, bundle, model_horizon, context_length, feature_cols, history_df

    def _predict_for_horizon(horizon: int) -> float:
        _model_path, bundle, model_horizon, _context_length, feature_cols, history_df = _load_bundle_for_horizon(
            horizon
        )
        preds = forecast_next_horizon(
            bundle.model,
            history_df,
            target_col="return",
            feature_cols=feature_cols,
            horizon=model_horizon,
        )
        preds = np.asarray(preds, dtype=float)
        return float(np.sum(preds[:model_horizon]))

    def _prefetch_horizons(horizons: list[int]) -> None:
        if not horizons:
            return
        reg_root = get_regression_models_dir()
        for horizon in horizons:
            model_path = reg_root / f"h{horizon:02d}"
            if not model_path.exists():
                model_path = model_dir
            _get_cached_bundle(model_path)

    use_parallel = os.getenv("PUMPFUN_PARALLEL_INFER", "0") == "1"
    first_batch = [h for h in horizons_needed if h <= 10]
    second_batch = [h for h in horizons_needed if h > 10]
    _prefetch_horizons(first_batch)

    if use_parallel and len(horizons_needed) > 1:
        import concurrent.futures

        max_workers = int(os.getenv("PUMPFUN_PARALLEL_WORKERS", "4"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_predict_for_horizon, h): h for h in horizons_needed}
            for future in concurrent.futures.as_completed(futures):
                horizon_predictions[futures[future]] = float(future.result())
    else:
        for horizon in first_batch:
            horizon_predictions[horizon] = _predict_for_horizon(horizon)
        if second_batch:
            _prefetch_horizons(second_batch)
            for horizon in second_batch:
                horizon_predictions[horizon] = _predict_for_horizon(horizon)

    results: list[MinutePrediction] = []
    for m in range(1, max_minutes + 1):
        horizon = min(m, horizon_cap)
        pred_return = horizon_predictions.get(horizon, 0.0)
        predicted_price = None
        price_change = None
        price_change_pct = None
        direction = classifier_direction or "FLAT"
        if current_price is not None and math.isfinite(pred_return):
            predicted_price = float(current_price * np.exp(pred_return))
            if math.isfinite(predicted_price):
                price_change = predicted_price - current_price
                if math.isfinite(price_change) and current_price:
                    price_change_pct = (price_change / current_price) * 100
                if classifier_direction is None and price_change is not None:
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

    result = PredictionResult(
        mint=mint,
        token_id=token_id,
        model_dir=str(model_dir),
        horizon=max_minutes,
        predictions=results,
    )
    
    # Clear model cache and force garbage collection to free memory after prediction
    # This ensures models are unloaded immediately after use
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    
    return result


_MODEL_CACHE: "OrderedDict[str, ModelBundle]" = OrderedDict()


def clear_model_cache() -> None:
    """Clear the model cache to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()


def unload_model(model_dir: Path) -> None:
    """Unload a specific model from cache to free memory."""
    global _MODEL_CACHE
    key = str(model_dir.resolve())
    if key in _MODEL_CACHE:
        del _MODEL_CACHE[key]
        gc.collect()


def _get_cached_bundle(model_dir: Path) -> ModelBundle:
    """Get model bundle from cache or load it. Cache size is limited to prevent memory issues."""
    key = str(model_dir.resolve())
    if key in _MODEL_CACHE:
        bundle = _MODEL_CACHE.pop(key)
        _MODEL_CACHE[key] = bundle
        return bundle
    # Clear cache before loading new model if cache is full (only keep 1 model)
    max_cache = int(os.getenv("PUMPFUN_MODEL_CACHE_SIZE", "1"))
    if len(_MODEL_CACHE) >= max_cache:
        # Clear all cached models before loading new one to free memory
        _MODEL_CACHE.clear()
        gc.collect()
    bundle = load_model_artifacts(model_dir)
    _MODEL_CACHE[key] = bundle
    return bundle
