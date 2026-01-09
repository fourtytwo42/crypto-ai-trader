"""Pump.fun backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pumpfun_train.forecast import forecast_next_horizon
from pumpfun_train.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.model_loader import load_model_artifacts


@dataclass
class PumpfunBacktestResult:
    mae: float
    rmse: float
    smape: float
    direction_accuracy: float
    price_accuracy_pct: float
    samples: int


def _load_holdout_tokens(model_dir: str | Path) -> list[str]:
    path = Path(model_dir) / "holdout_tokens.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def backtest_pumpfun_model(
    model_dir: str | Path,
    minutes: int = 10,
    test_window: int = 240,
    target_mode: str = "sum",
    max_tokens: int | None = None,
    max_samples: int | None = None,
    use_reserve_tokens: bool = False,
    show_progress: bool = True,  # Allow disabling progress bar for nested contexts
    progress_position: int | None = None,
    sample_stride: int = 1,
) -> PumpfunBacktestResult:
    def _status_write(message: str) -> None:
        try:
            from tqdm import tqdm
            tqdm.write(message)
        except Exception:
            print(message)

    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    bundle = load_model_artifacts(model_dir)
    config_meta = bundle.metadata.get("config", {})
    model_horizon = int(config_meta.get("horizon", 10))
    context_length = int(config_meta.get("context_length", 336))

    if target_mode == "sum" and minutes > model_horizon:
        raise ValueError(
            f"Requested {minutes} minutes, but model horizon is {model_horizon}. Retrain with a larger horizon."
        )

    db = get_pumpfun_db_manager()
    
    # Load test tokens (outside session to avoid holding it while loading data)
    if use_reserve_tokens:
        # Use reserve tokens (completely separate from training/holdout)
        from pumpfun_train.reserve_tokens import get_reserve_tokens
        from pumpfun_train.reserve_cache import load_reserve_cache, save_reserve_cache
        
        test_tokens = get_reserve_tokens()
        if not test_tokens:
            raise ValueError("No reserve tokens available. Run reserve token selection first.")
        
        if max_tokens is not None:
            test_tokens = test_tokens[:max_tokens]
        
        # Try to load from cache first (cache contains all reserve tokens)
        cached_training = load_reserve_cache()
        if cached_training is not None and not cached_training.df.empty:
            # Filter cached data to requested tokens
            cached_df = cached_training.df
            cached_normalizers = cached_training.normalizers
            
            # Check if all requested tokens are in cache
            cached_token_set = set(cached_df["token_id"].unique())
            requested_token_set = set(test_tokens)
            
            if requested_token_set.issubset(cached_token_set):
                # All requested tokens are in cache, filter to them
                filtered_df = cached_df[cached_df["token_id"].isin(test_tokens)].copy()
                filtered_normalizers = {tid: cached_normalizers[tid] for tid in test_tokens if tid in cached_normalizers}
                from pumpfun_train.data import PumpfunTrainingData
                training = PumpfunTrainingData(df=filtered_df, normalizers=filtered_normalizers)
                # Cache hit - no need to load from DB (suppress "Loading token data" message)
                _status_write(
                    f"✓ Using cached reserve token data ({len(cached_df):,} rows, {len(cached_token_set)} tokens)"
                )
            else:
                # Need to load fresh (some tokens missing from cache)
                # This should only happen if reserve tokens were changed
                with db.session() as session:
                    training = prepare_pumpfun_training_data(session, test_tokens, normalize=True)
                    # Save full cache for next time (with ALL reserve tokens)
                    if not training.df.empty:
                        # Load all reserve tokens for cache (not just requested subset)
                        all_reserve_tokens = get_reserve_tokens()
                        if all_reserve_tokens:
                            full_training = prepare_pumpfun_training_data(session, all_reserve_tokens, normalize=True)
                            if not full_training.df.empty:
                                save_reserve_cache(full_training)
        else:
            # No cache, load fresh and save
            with db.session() as session:
                # Load all reserve tokens for cache (not just requested subset)
                all_reserve_tokens = get_reserve_tokens()
                if all_reserve_tokens:
                    # Load all reserve tokens once, cache them
                    full_training = prepare_pumpfun_training_data(session, all_reserve_tokens, normalize=True)
                    if not full_training.df.empty:
                        save_reserve_cache(full_training)
                        # Filter to requested tokens
                        filtered_df = full_training.df[full_training.df["token_id"].isin(test_tokens)].copy()
                        filtered_normalizers = {tid: full_training.normalizers[tid] for tid in test_tokens if tid in full_training.normalizers}
                        from pumpfun_train.data import PumpfunTrainingData
                        training = PumpfunTrainingData(df=filtered_df, normalizers=filtered_normalizers)
                    else:
                        training = prepare_pumpfun_training_data(session, test_tokens, normalize=True)
                else:
                    training = prepare_pumpfun_training_data(session, test_tokens, normalize=True)
    else:
        # Use holdout tokens (default behavior)
        test_tokens = _load_holdout_tokens(model_dir)
        if not test_tokens:
            raise ValueError("No holdout tokens found for backtest")
        
        if max_tokens is not None:
            test_tokens = test_tokens[:max_tokens]
        
        # Load training data for holdout tokens (no cache for holdout)
        with db.session() as session:
            training = prepare_pumpfun_training_data(session, test_tokens, normalize=True)

    df = training.df
    if df.empty:
        raise ValueError("No holdout data available for backtest")

    if target_mode == "direct":
        df = add_return_target(df, minutes)
    df = df.dropna().reset_index(drop=True)

    preds_all = []
    actual_all = []
    dir_hits = []
    price_acc = []

    feature_cols = [col for col in [*PUMPFUN_FEATURE_COLUMNS, "log_close", "log_volume"] if col in df.columns]

    # Group by token once
    token_groups = list(df.groupby("token_id"))
    
    # Parallel backtesting using ThreadPoolExecutor
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x
    
    # Determine number of workers (use threads for I/O-bound model predictions)
    max_workers = min(os.cpu_count() or 4, len(token_groups), 8)  # Cap at 8 to avoid memory issues
    
    # Shared results list with lock for thread safety
    results_lock = Lock()
    tokens_processed = 0
    tokens_skipped = 0
    min_required_rows = context_length + minutes
    
    # Worker function to backtest a single token
    def _backtest_single_token(args: tuple) -> tuple[str, list, list, list, list, int, bool]:
        """Backtest a single token and return results."""
        token_id, token_df, bundle_model, feature_cols_local, target_mode_local, minutes_local, model_horizon_local, context_length_local, test_window_local, max_samples_local, min_required_rows_local, sample_stride_local = args
        
        token_df = token_df.sort_values("timestamp").reset_index(drop=True)
        if len(token_df) < min_required_rows_local:
            return (token_id, [], [], [], [], 0, True)  # (token_id, preds, actuals, dir_hits, price_acc, samples, skipped=True)
        
        token_preds = []
        token_actuals = []
        token_dir_hits = []
        token_price_acc = []
        
        start_idx = max(context_length_local, len(token_df) - test_window_local)
        for idx in range(start_idx, len(token_df) - minutes_local, sample_stride_local):
            history = token_df.iloc[idx - context_length_local : idx].reset_index(drop=True)
            preds = forecast_next_horizon(
                bundle_model,
                history,
                target_col="return_horizon" if target_mode_local == "direct" else "return",
                feature_cols=feature_cols_local,
                horizon=model_horizon_local,
            )
            preds = np.asarray(preds, dtype=float)
            if target_mode_local == "direct":
                pred_return = float(preds[-1])
                actual_return = float(token_df["return_horizon"].iloc[idx])
            else:
                pred_return = float(np.sum(preds[:minutes_local]))
                actual_return = float(token_df["return"].iloc[idx : idx + minutes_local].sum())
            
            token_preds.append(pred_return)
            token_actuals.append(actual_return)
            token_dir_hits.append(int(np.sign(pred_return) == np.sign(actual_return)))
            
            if "raw_log_close" in history.columns and "raw_log_close" in token_df.columns:
                last_raw_log_close = float(history["raw_log_close"].iloc[-1])
                actual_raw_log_close = float(token_df["raw_log_close"].iloc[idx + minutes_local])
                pred_raw_log_close = last_raw_log_close + pred_return
                pred_close = float(np.exp(pred_raw_log_close))
                actual_close = float(np.exp(actual_raw_log_close))
                if actual_close == 0:
                    token_price_acc.append(float("nan"))
                else:
                    token_price_acc.append(max(0.0, 1.0 - abs(pred_close - actual_close) / actual_close) * 100.0)
            
            # No per-token limit - process all available samples for each token
            # (max_samples is handled at the aggregate level if needed)
        
        return (token_id, token_preds, token_actuals, token_dir_hits, token_price_acc, len(token_preds), False)
    
    # Prepare worker arguments
    worker_args = [
        (
            token_id,
            token_df,
            bundle.model,  # Model is shared (PyTorch models are thread-safe for inference)
            feature_cols,
            target_mode,
            minutes,
            model_horizon,
            context_length,
            test_window,
            max_samples,
            min_required_rows,  # Pass min_required_rows to worker
            sample_stride,
        )
        for token_id, token_df in token_groups
    ]
    
    # Progress bar for parallel processing
    # Disable if show_progress=False (for nested contexts to avoid conflicts)
    if show_progress:
        try:
            from tqdm import tqdm
            token_pbar = tqdm(
                total=len(token_groups),
                desc=f"Backtesting tokens ({max_workers} workers)",
                unit="token",
                leave=False,  # Don't leave bar when done (nested bar)
                ncols=100,
                mininterval=0.1,  # Update frequently for visibility
                miniters=1,  # Update after each token
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                dynamic_ncols=True,  # Adjust to terminal width
                position=progress_position,
            )
        except ImportError:
            token_pbar = None
    else:
        # Dummy progress bar that does nothing (for nested contexts)
        class DummyPbar:
            def update(self, n=1): pass
            def set_postfix(self, **kwargs): pass
            def close(self): pass
        token_pbar = DummyPbar()
    
    # Process tokens in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_token = {
            executor.submit(_backtest_single_token, args): args[0]
            for args in worker_args
        }
        
        # Process completed tasks
        for future in as_completed(future_to_token):
            try:
                token_id, token_preds, token_actuals, token_dir_hits, token_price_acc, token_samples, was_skipped = future.result()
                
                with results_lock:
                    if was_skipped:
                        tokens_skipped += 1
                    else:
                        tokens_processed += 1
                        preds_all.extend(token_preds)
                        actual_all.extend(token_actuals)
                        dir_hits.extend(token_dir_hits)
                        price_acc.extend(token_price_acc)
                
                # Update progress bar
                token_pbar.update(1)
                token_pbar.set_postfix({
                    "samples": len(preds_all),
                    "processed": tokens_processed,
                    "skipped": tokens_skipped
                })
            except Exception as e:
                # Log error but continue processing
                with results_lock:
                    tokens_skipped += 1
                    current_samples = len(preds_all)
                    current_processed = tokens_processed
                    current_skipped = tokens_skipped
                
                if token_pbar is not None:
                    token_pbar.update(1)
                    token_pbar.set_postfix({
                        "samples": current_samples,
                        "processed": current_processed,
                        "skipped": current_skipped,
                        "error": str(e)[:20]
                    })
    
    if token_pbar is not None:
        token_pbar.close()
    
    preds_arr = np.asarray(preds_all, dtype=float)
    actual_arr = np.asarray(actual_all, dtype=float)
    if preds_arr.size == 0:
        raise ValueError(
            f"No backtest samples generated. Processed {tokens_processed} tokens, "
            f"skipped {tokens_skipped} tokens (insufficient data: need {context_length + minutes} rows). "
            f"Total tokens in test set: {len(df.groupby('token_id'))}"
        )

    mae = float(np.mean(np.abs(preds_arr - actual_arr)))
    rmse = float(np.sqrt(np.mean((preds_arr - actual_arr) ** 2)))
    denom = np.abs(actual_arr) + np.abs(preds_arr) + 1e-8
    smape = float(np.mean(2.0 * np.abs(preds_arr - actual_arr) / denom)) * 100.0
    direction_accuracy = float(np.mean(dir_hits)) * 100.0
    price_accuracy_pct = float(np.nanmean(price_acc)) if price_acc else float("nan")

    return PumpfunBacktestResult(
        mae=mae,
        rmse=rmse,
        smape=smape,
        direction_accuracy=direction_accuracy,
        price_accuracy_pct=price_accuracy_pct,
        samples=int(preds_arr.size),
    )
