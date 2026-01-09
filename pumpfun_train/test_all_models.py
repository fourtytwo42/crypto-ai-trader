#!/usr/bin/env python3
"""Test all 20 regression models and report accuracy metrics.

Runs backtests on all trained regression models (h01-h20) using holdout tokens
and displays comprehensive accuracy metrics.
"""

from __future__ import annotations

import sys
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
import argparse

# Suppress logging noise during testing
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
os.environ["LOG_LEVEL"] = "CRITICAL"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pumpfun_train.cli.commands import pumpfun_backtest_command


def format_metric(value: float, is_percent: bool = False, decimals: int = 2) -> str:
    """Format a metric value for display."""
    if is_percent:
        return f"{value:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def main() -> None:
    """Test all 20 regression models."""
    parser = argparse.ArgumentParser(description="Test all 20 regression models.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use faster backtest defaults for reserve tokens (fewer tokens, smaller window, stride > 1).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    models_dir = base_dir / "models" / "regression"
    
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        print("Please train models first using train_regression_models.py")
        return
    
    print("="*80)
    print("TESTING ALL 20 PRICE PREDICTION MODELS")
    print("="*80)
    print("Choose evaluation tokens:")
    print("  1. Holdout tokens (used during training validation, default)")
    print("  2. Reserve tokens (completely separate, never seen during training)")
    print()
    
    # Check if reserve tokens exist
    from pumpfun_train.reserve_tokens import get_reserve_tokens, RESERVE_TOKEN_FILE
    try:
        reserve_tokens = get_reserve_tokens()
        reserve_count = len(reserve_tokens)
    except Exception:
        reserve_tokens = []
        reserve_count = 0
    
    if reserve_count == 0:
        print("⚠ No reserve tokens found. Using holdout tokens only.")
        print("   Run 'python3 -m pumpfun_train.cli_main pumpfun-select-reserve' to create reserve tokens")
        use_reserve = False
    else:
        choice = input(f"Enter choice (1 or 2, default=1): ").strip()
        use_reserve = choice == "2"
    
    if use_reserve:
        print(f"Using reserve tokens (completely separate from training)...")
        print(f"✓ Found {reserve_count} reserve tokens")
        print("Note: First model will load data from database (may take ~2 min),")
        print("      subsequent models will use cache (much faster).")
    else:
        print("Using holdout tokens (reserved during training)...")
    print("This may take several minutes...")
    print()

    test_window = 240
    max_tokens = None
    sample_stride = 1
    if use_reserve and args.fast:
        test_window = 120
        max_tokens = 20
        sample_stride = 2
        print("Fast backtest enabled for reserve tokens:")
        print(f"  - test_window={test_window}, max_tokens={max_tokens}, sample_stride={sample_stride}")
        print()
    
    horizons = list(range(1, 21))
    results = []
    
    # Progress bar for all models (outer bar at position=0)
    test_pbar = tqdm(
        horizons,
        desc="Testing models",
        unit="model",
        ncols=100,
        position=0,  # Outer bar at position 0
        leave=True,  # Keep outer bar visible
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}"
    )
    
    for horizon in test_pbar:
        horizon_dir = models_dir / f"h{horizon:02d}"
        test_pbar.set_description(f"Testing h{horizon:02d} ({horizon}min)")
        
        if not horizon_dir.exists():
            test_pbar.set_postfix({"status": "NOT FOUND"})
            results.append({
                "horizon": horizon,
                "status": "not_found",
                "error": "Model directory not found"
            })
            continue
        
        start_time = time.time()
        try:
            result = pumpfun_backtest_command(
                model_dir=str(horizon_dir),
                minutes=horizon,
                test_window=test_window,
                target_mode="sum",
                max_tokens=max_tokens,
                max_samples=None,  # No limit - test all available samples for accurate metrics
                use_reserve_tokens=use_reserve,
                show_progress=True,  # Show per-token progress so long runs don't look hung
                progress_position=1,
                sample_stride=sample_stride,
            )
            elapsed = time.time() - start_time
            
            results.append({
                "horizon": horizon,
                "status": "success",
                "mae": result.get("mae", 0),
                "rmse": result.get("rmse", 0),
                "smape": result.get("smape", 0),
                "direction_accuracy": result.get("direction_accuracy", 0),
                "price_accuracy_pct": result.get("price_accuracy_pct", 0),
                "samples": result.get("samples", 0),
                "time": elapsed,
            })
            
            # Update progress bar with key metrics
            dir_acc = result.get("direction_accuracy", 0)
            price_acc = result.get("price_accuracy_pct", 0)
            test_pbar.set_postfix({
                "dir_acc": f"{dir_acc:.1f}%",
                "price_acc": f"{price_acc:.1f}%",
                "samples": result.get("samples", 0)
            })
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            # Truncate very long error messages for display
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            test_pbar.set_postfix({"status": "ERROR", "error": error_msg[:30]})
            print(f"\n⚠ Error testing h{horizon:02d}: {error_msg}")
            results.append({
                "horizon": horizon,
                "status": "error",
                "error": str(e),
                "time": elapsed,
            })
    
    test_pbar.close()
    print()
    
    # Display results summary
    print("="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print()
    
    # Table header
    token_type = "Reserve" if use_reserve else "Holdout"
    reserve_count = len(reserve_tokens) if use_reserve else 0
    token_count_display = reserve_count
    if use_reserve and max_tokens is not None:
        token_count_display = min(reserve_count, max_tokens)
    print(f"{'Horizon':<10} {'Direction':<12} {'Price Acc':<12} {'MAE':<12} {'RMSE':<12} {'SMAPE':<12} {'Samples':<10} {'Status'}")
    print(f"(Testing on {token_type} tokens - {token_count_display if use_reserve else '12'} tokens available)")
    print("-"*80)
    
    successful_results = [r for r in results if r.get("status") == "success"]
    failed_results = [r for r in results if r.get("status") != "success"]
    
    # Display successful results
    for result in sorted(successful_results, key=lambda x: x["horizon"]):
        horizon = result["horizon"]
        dir_acc = result.get("direction_accuracy", 0)
        price_acc = result.get("price_accuracy_pct", 0)
        mae = result.get("mae", 0)
        rmse = result.get("rmse", 0)
        smape = result.get("smape", 0)
        samples = result.get("samples", 0)
        
        print(
            f"{horizon:>3} min   "
            f"{format_metric(dir_acc, is_percent=True):<12} "
            f"{format_metric(price_acc, is_percent=True):<12} "
            f"{format_metric(mae, decimals=6):<12} "
            f"{format_metric(rmse, decimals=6):<12} "
            f"{format_metric(smape, is_percent=True):<12} "
            f"{samples:>8}   "
            f"✓"
        )
    
    # Display failed results
    for result in sorted(failed_results, key=lambda x: x["horizon"]):
        horizon = result["horizon"]
        status = result.get("status", "unknown")
        error = result.get("error", "Unknown error")
        print(
            f"{horizon:>3} min   "
            f"{'N/A':<12} "
            f"{'N/A':<12} "
            f"{'N/A':<12} "
            f"{'N/A':<12} "
            f"{'N/A':<12} "
            f"{'N/A':<10} "
            f"✗ {status}: {error}"
        )
    
    print()
    print("="*80)
    print("METRICS EXPLANATION")
    print("="*80)
    print()
    print("Direction Accuracy: % of predictions that correctly predict price direction (up/down)")
    print("Price Accuracy %:   % accuracy of predicted price vs actual price")
    print("MAE (Mean Absolute Error): Average absolute difference between predicted and actual returns")
    print("RMSE (Root Mean Squared Error): Square root of average squared errors (penalizes large errors)")
    print("SMAPE (Symmetric MAPE): Symmetric mean absolute percentage error in %")
    print("Samples: Number of predictions tested")
    token_type = "Reserve" if use_reserve else "Holdout"
    reserve_count = len(reserve_tokens) if use_reserve else 0
    print(f"Tokens used: {token_type} tokens ({reserve_count if use_reserve else '12'} tokens)")
    print()
    
    if successful_results:
        # Calculate averages
        avg_dir_acc = sum(r.get("direction_accuracy", 0) for r in successful_results) / len(successful_results)
        avg_price_acc = sum(r.get("price_accuracy_pct", 0) for r in successful_results if r.get("price_accuracy_pct", 0) > 0) / max(1, sum(1 for r in successful_results if r.get("price_accuracy_pct", 0) > 0))
        avg_mae = sum(r.get("mae", 0) for r in successful_results) / len(successful_results)
        avg_rmse = sum(r.get("rmse", 0) for r in successful_results) / len(successful_results)
        avg_smape = sum(r.get("smape", 0) for r in successful_results) / len(successful_results)
        total_samples = sum(r.get("samples", 0) for r in successful_results)
        total_time = sum(r.get("time", 0) for r in successful_results)
        
        print("="*80)
        print("AVERAGE METRICS (across all 20 models)")
        print("="*80)
        print(f"Average Direction Accuracy: {format_metric(avg_dir_acc, is_percent=True)}")
        print(f"Average Price Accuracy:     {format_metric(avg_price_acc, is_percent=True)}")
        print(f"Average MAE:                {format_metric(avg_mae, decimals=6)}")
        print(f"Average RMSE:               {format_metric(avg_rmse, decimals=6)}")
        print(f"Average SMAPE:              {format_metric(avg_smape, is_percent=True)}")
        print(f"Total Samples Tested:       {total_samples:,}")
        print(f"Total Testing Time:         {total_time:.1f}s")
        print()
        
        # Find best and worst models
        best_dir = max(successful_results, key=lambda x: x.get("direction_accuracy", 0))
        worst_dir = min(successful_results, key=lambda x: x.get("direction_accuracy", 0))
        best_price = max((r for r in successful_results if r.get("price_accuracy_pct", 0) > 0), key=lambda x: x.get("price_accuracy_pct", 0), default=None)
        worst_price = min((r for r in successful_results if r.get("price_accuracy_pct", 0) > 0), key=lambda x: x.get("price_accuracy_pct", 0), default=None)
        
        print("="*80)
        print("BEST/WORST MODELS")
        print("="*80)
        print(f"Best Direction Accuracy:   h{best_dir['horizon']:02d} ({best_dir['horizon']} min) - {format_metric(best_dir.get('direction_accuracy', 0), is_percent=True)}")
        print(f"Worst Direction Accuracy:  h{worst_dir['horizon']:02d} ({worst_dir['horizon']} min) - {format_metric(worst_dir.get('direction_accuracy', 0), is_percent=True)}")
        if best_price:
            print(f"Best Price Accuracy:       h{best_price['horizon']:02d} ({best_price['horizon']} min) - {format_metric(best_price.get('price_accuracy_pct', 0), is_percent=True)}")
        if worst_price:
            print(f"Worst Price Accuracy:      h{worst_price['horizon']:02d} ({worst_price['horizon']} min) - {format_metric(worst_price.get('price_accuracy_pct', 0), is_percent=True)}")
        print()


if __name__ == "__main__":
    main()
