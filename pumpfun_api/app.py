from __future__ import annotations

import logging
import time
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from predictor import get_candles, get_token_snapshot, predict_minutes, clear_model_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pump.fun Prediction API")

# Thread pool for running blocking prediction operations
_prediction_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down prediction executor...")
    _prediction_executor.shutdown(wait=True, timeout=5)
    clear_model_cache()
    logger.info("Shutdown complete")

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and their response times."""
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path} - Query: {dict(request.query_params)}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Error: {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s", exc_info=True)
        raise


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cache/clear")
def clear_cache() -> dict[str, str]:
    """Clear the model cache to free memory."""
    try:
        clear_model_cache()
        logger.info("Model cache cleared")
        return {"status": "ok", "message": "Model cache cleared"}
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/memory")
def memory_status() -> dict[str, object]:
    """Get current memory usage statistics."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        return {
            "rss_mb": round(mem_info.rss / 1024 / 1024, 2),
            "vms_mb": round(mem_info.vms / 1024 / 1024, 2),
            "percent": round(mem_percent, 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "total_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
        }
    else:
        return {"status": "psutil not available", "message": "Install psutil for memory monitoring"}

@app.get("/token")
def token(mint: str = Query(...)) -> dict[str, object]:
    """Get token snapshot information."""
    try:
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
    except ValueError as e:
        logger.warning(f"Token not found: mint={mint}, error={str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Token error: mint={mint}, error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/candles")
def candles(mint: str = Query(...), limit: int = Query(240, ge=10, le=2000)) -> dict[str, object]:
    """Get candle data for a token."""
    try:
        series = get_candles(mint, limit=limit)
        return {"token_id": series.token_id, "candles": series.candles}
    except ValueError as e:
        logger.warning(f"Candles not found: mint={mint}, error={str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Candles error: mint={mint}, error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/predict")
async def predict(mint: str = Query(...), minutes: int = Query(10, ge=1, le=60)) -> dict[str, object]:
    """Get price predictions for a token at various time horizons."""
    try:
        logger.info(f"Predict request: mint={mint}, minutes={minutes}")
        
        # Run blocking prediction in thread pool with timeout (60 seconds)
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_prediction_executor, predict_minutes, mint, minutes),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Predict timeout: mint={mint}, minutes={minutes} - took longer than 60s")
            raise HTTPException(status_code=504, detail="Prediction request timed out after 60 seconds")
        except FutureTimeoutError:
            logger.error(f"Predict timeout: mint={mint}, minutes={minutes}")
            raise HTTPException(status_code=504, detail="Prediction request timed out")
        
        logger.info(f"Predict success: mint={mint}, predictions={len(result.predictions)}")
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
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Predict validation error: mint={mint}, error={str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Predict error: mint={mint}, error={str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
