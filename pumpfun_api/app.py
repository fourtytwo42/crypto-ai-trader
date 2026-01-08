from __future__ import annotations

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from predictor import get_candles, get_token_snapshot, predict_minutes

app = FastAPI(title="Pump.fun Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/token")
def token(mint: str = Query(...)) -> dict[str, object]:
    snapshot = get_token_snapshot(mint)
    return {
        "mint": snapshot.mint,
        "token_id": snapshot.token_id,
        "symbol": snapshot.symbol,
        "name": snapshot.name,
        "created_timestamp": snapshot.created_timestamp,
        "king_of_the_hill_timestamp": snapshot.king_of_the_hill_timestamp,
        "completed": snapshot.completed,
        "current_price": snapshot.current_price,
        "market_cap_usd": snapshot.market_cap_usd,
        "last_trade_timestamp": snapshot.last_trade_timestamp,
    }


@app.get("/candles")
def candles(mint: str = Query(...), limit: int = Query(240, ge=10, le=2000)) -> dict[str, object]:
    series = get_candles(mint, limit=limit)
    return {"token_id": series.token_id, "candles": series.candles}


@app.get("/predict")
def predict(mint: str = Query(...), minutes: int = Query(10, ge=1, le=60)) -> dict[str, object]:
    result = predict_minutes(mint, minutes)
    return {
        "mint": result.mint,
        "token_id": result.token_id,
        "model_dir": result.model_dir,
        "horizon": result.horizon,
        "predictions": [
            {
                "minutes": item.minutes,
                "direction": item.direction,
                "current_price": item.current_price,
                "predicted_price": item.predicted_price,
                "price_change": item.price_change,
                "price_change_pct": item.price_change_pct,
                "direction_confidence": item.direction_confidence,
            }
            for item in result.predictions
        ],
    }
