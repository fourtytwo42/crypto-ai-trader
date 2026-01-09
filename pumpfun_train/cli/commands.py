"""CLI command implementations for Pump.fun operations."""

from __future__ import annotations

import structlog

from pumpfun_train.backtest import backtest_pumpfun_model
from pumpfun_train.reserve_tokens import refresh_reserve_tokens
from pumpfun_train.classifier import (
    backtest_pumpfun_direction_classifier,
    train_pumpfun_direction_classifier,
)
from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.pipeline import PumpfunFilterConfig, sync_pumpfun_candles
from pumpfun_train.predict import predict_pumpfun
from pumpfun_train.training import train_pumpfun_model

logger = structlog.get_logger(__name__)


def pumpfun_sync_command(
    min_age_minutes: int = 60,
    active_age_minutes: int = 240,
    min_total_trades: int = 10,
    min_recent_trades: int = 10,
    recent_window_minutes: int = 30,
    replace_existing: bool = False,
    max_tokens: int | None = None,
    price_lookup_enabled: bool = True,
    max_workers: int | None = None,
) -> dict[str, int]:
    """Sync pump.fun trades into minute candles/features with parallel processing."""
    from pumpfun_train.training_cache import invalidate_cache
    
    config = PumpfunFilterConfig(
        min_age_minutes=min_age_minutes,
        active_age_minutes=active_age_minutes,
        min_total_trades=min_total_trades,
        min_recent_trades=min_recent_trades,
        recent_window_minutes=recent_window_minutes,
    )
    db = get_pumpfun_db_manager()
    with db.session() as session:
        result = sync_pumpfun_candles(
            session,
            config,
            replace_existing=replace_existing,
            max_tokens=max_tokens,
            price_lookup_enabled=price_lookup_enabled,
            max_workers=max_workers,
        )
    
    # If we added new data, invalidate the cache
    if result.get("features", 0) > 0:
        invalidate_cache()
        print("✓ Training data cache invalidated (database was updated)")
    
    return result


def pumpfun_train_command(
    model_dir: str,
    horizon_minutes: int = 10,
    context_length: int = 336,
    model_type: str = "nhits",
    target_mode: str = "sum",
    hidden_size: int = 512,
    num_layers: int = 3,
    patch_length: int = 8,
    stride: int = 4,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    holdout_count: int = 12,
) -> dict[str, object]:
    """Train pump.fun model on minute candles."""
    result = train_pumpfun_model(
        model_dir=model_dir,
        horizon_minutes=horizon_minutes,
        context_length=context_length,
        model_type=model_type,
        target_mode=target_mode,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        holdout_count=holdout_count,
    )
    return {
        "model_dir": str(result.model_dir),
        "metrics": result.metrics,
        "holdout_tokens": result.holdout_tokens,
    }


def pumpfun_backtest_command(
    model_dir: str,
    minutes: int = 10,
    test_window: int = 240,
    target_mode: str = "sum",
    max_tokens: int | None = None,
    max_samples: int | None = None,
    use_reserve_tokens: bool = False,
    show_progress: bool = True,  # Allow disabling progress bar for nested contexts
    progress_position: int | None = None,
    sample_stride: int = 1,
) -> dict[str, object]:
    """Backtest pump.fun model on holdout or reserve tokens.
    
    Args:
        model_dir: Model directory path
        minutes: Prediction horizon in minutes
        test_window: Number of recent rows to use for testing
        target_mode: 'sum' or 'direct'
        max_tokens: Maximum number of tokens to test
        max_samples: Maximum number of samples to test
        use_reserve_tokens: If True, use reserve tokens (completely separate from training),
                          otherwise use holdout tokens (default)
    """
    result = backtest_pumpfun_model(
        model_dir=model_dir,
        minutes=minutes,
        test_window=test_window,
        target_mode=target_mode,
        max_tokens=max_tokens,
        max_samples=max_samples,
        use_reserve_tokens=use_reserve_tokens,
        show_progress=show_progress,
        progress_position=progress_position,
        sample_stride=sample_stride,
    )
    return {
        "mae": result.mae,
        "rmse": result.rmse,
        "smape": result.smape,
        "direction_accuracy": result.direction_accuracy,
        "price_accuracy_pct": result.price_accuracy_pct,
        "samples": result.samples,
    }


def pumpfun_predict_command(
    token_id: str,
    model_dir: str,
    minutes: int = 10,
    target_mode: str = "sum",
) -> dict[str, object]:
    """Predict pump.fun token price movement."""
    result = predict_pumpfun(
        token_id=token_id, model_dir=model_dir, minutes=minutes, target_mode=target_mode
    )
    return {
        "token_id": result.token_id,
        "current_price": result.current_price,
        "predicted_price": result.predicted_price,
        "price_change": result.price_change,
        "price_change_pct": result.price_change_pct,
        "direction": result.direction,
        "minutes": result.minutes,
        "confidence": result.confidence,
    }


def pumpfun_classify_train_command(
    model_dir: str,
    horizon_minutes: int = 10,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    label_threshold: float = 0.0,
    holdout_count: int = 12,
    min_token_samples: int = 0,
    use_pos_weight: bool = True,
    normalize_features: bool = True,
    max_samples: int | None = 10_000_000,
) -> dict[str, object]:
    """Train pump.fun direction classifier.
    
    Args:
        max_samples: Maximum samples to load (default: 10M to prevent OOM)
    """
    result = train_pumpfun_direction_classifier(
        model_dir=model_dir,
        horizon_minutes=horizon_minutes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        label_threshold=label_threshold,
        holdout_count=holdout_count,
        min_token_samples=min_token_samples,
        use_pos_weight=use_pos_weight,
        normalize_features=normalize_features,
        max_samples=max_samples,
    )
    return {
        "model_dir": str(result.model_dir),
        "metrics": result.metrics,
        "holdout_tokens": result.holdout_tokens,
    }


def pumpfun_classify_backtest_command(
    model_dir: str,
    max_tokens: int | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Backtest pump.fun direction classifier."""
    return backtest_pumpfun_direction_classifier(
        model_dir=model_dir, max_tokens=max_tokens, max_samples=max_samples
    )


def pumpfun_clear_processed_command() -> dict[str, int]:
    """Clear all processed candle and feature data (keeps original trades/tokens).
    
    This is safe to run - it only clears derived data, not the original pump.fun trade data.
    After clearing, run sync to regenerate candles/features.
    
    Uses TRUNCATE for fast deletion - much faster than DELETE for large tables.
    """
    from pumpfun_train.db import get_pumpfun_db_manager
    from pumpfun_train.models import PumpCandle1m, PumpFeature1m, PumpBase
    from pumpfun_train.training_cache import invalidate_cache
    from sqlalchemy import text
    
    db = get_pumpfun_db_manager()
    with db.session() as session:
        # Use TRUNCATE which is much faster than DELETE for large tables
        # It's instant and doesn't need to count rows
        print("Truncating pump_candles_1m table...", end="", flush=True)
        session.execute(text("TRUNCATE TABLE pump_candles_1m RESTART IDENTITY CASCADE"))
        print(" done", flush=True)
        
        print("Truncating pump_features_1m table...", end="", flush=True)
        session.execute(text("TRUNCATE TABLE pump_features_1m RESTART IDENTITY CASCADE"))
        print(" done", flush=True)
        
        print("Committing changes...", end="", flush=True)
        session.commit()
        print(" done", flush=True)
    
    # Invalidate training data cache since database was updated
    invalidate_cache()
    print("✓ Training data cache invalidated (will reload on next training run)")
    
    # Return success (we don't count because it's slow)
    return {
        "candles_deleted": -1,  # -1 indicates "all" without counting
        "features_deleted": -1,
    }


def pumpfun_select_reserve_command(reserve_count: int = 50) -> dict[str, object]:
    """Select reserve tokens for evaluation (completely separate from training/holdout)."""
    reserve_tokens = refresh_reserve_tokens(reserve_count=reserve_count)
    return {
        "reserve_tokens": reserve_tokens,
        "count": len(reserve_tokens),
    }
