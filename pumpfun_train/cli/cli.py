"""CLI command interface for Pump.fun Prediction Model.

Provides Click-based command-line interface for pump.fun operations.
"""

import click
import structlog

from pumpfun_train.cli.commands import (
    pumpfun_backtest_command,
    pumpfun_classify_backtest_command,
    pumpfun_classify_train_command,
    pumpfun_clear_processed_command,
    pumpfun_predict_command,
    pumpfun_select_reserve_command,
    pumpfun_sync_command,
    pumpfun_train_command,
)

logger = structlog.get_logger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Pump.fun Prediction Model CLI.

    Commands for training, predicting, and managing pump.fun models.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("pumpfun-sync")
@click.option("--min-age-minutes", type=int, default=60)
@click.option("--active-age-minutes", type=int, default=240)
@click.option("--min-total-trades", type=int, default=10)
@click.option("--min-recent-trades", type=int, default=10)
@click.option("--recent-window-minutes", type=int, default=30)
@click.option("--max-tokens", type=int, default=None)
@click.option("--skip-price-lookup", is_flag=True, help="Skip SOL/USD lookup for missing USD fields")
@click.option("--replace", is_flag=True, help="Replace existing candles/features")
@click.option("--max-workers", type=int, default=None, help="Number of parallel workers (default: CPU count, max 8)")
def pumpfun_sync(
    min_age_minutes: int,
    active_age_minutes: int,
    min_total_trades: int,
    min_recent_trades: int,
    recent_window_minutes: int,
    max_tokens: int | None,
    skip_price_lookup: bool,
    replace: bool,
    max_workers: int | None,
) -> None:
    """Sync pump.fun trades into minute candles/features with parallel processing."""
    result = pumpfun_sync_command(
        min_age_minutes=min_age_minutes,
        active_age_minutes=active_age_minutes,
        min_total_trades=min_total_trades,
        min_recent_trades=min_recent_trades,
        recent_window_minutes=recent_window_minutes,
        replace_existing=replace,
        max_tokens=max_tokens,
        price_lookup_enabled=not skip_price_lookup,
        max_workers=max_workers,
    )
    click.echo(
        f"Pump.fun sync complete: tokens={result['tokens']} candles={result['candles']} features={result['features']}"
    )


@cli.command("pumpfun-train")
@click.option("--model-dir", default="models_pumpfun_nhits", help="Directory to save model")
@click.option("--horizon-minutes", type=int, default=10)
@click.option("--context-length", type=int, default=336)
@click.option("--model-type", type=click.Choice(["nhits", "patchtst"]), default="nhits")
@click.option("--target-mode", type=click.Choice(["sum", "direct"]), default="sum")
@click.option("--hidden-size", type=int, default=512)
@click.option("--num-layers", type=int, default=3)
@click.option("--patch-length", type=int, default=8)
@click.option("--stride", type=int, default=4)
@click.option("--epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=16)
@click.option("--learning-rate", type=float, default=5e-5)
@click.option("--holdout-count", type=int, default=12)
def pumpfun_train(
    model_dir: str,
    horizon_minutes: int,
    context_length: int,
    model_type: str,
    target_mode: str,
    hidden_size: int,
    num_layers: int,
    patch_length: int,
    stride: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    holdout_count: int,
) -> None:
    """Train pump.fun minute model."""
    result = pumpfun_train_command(
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
    click.echo(f"Pump.fun training complete. Metrics: {result['metrics']}")
    click.echo(f"Holdout tokens saved: {len(result['holdout_tokens'])}")


@cli.command("pumpfun-backtest")
@click.option("--model-dir", required=True, help="Directory to load model")
@click.option("--minutes", type=int, default=10)
@click.option("--test-window", type=int, default=240)
@click.option("--target-mode", type=click.Choice(["sum", "direct"]), default="sum")
@click.option("--max-tokens", type=int, default=None)
@click.option("--max-samples", type=int, default=None)
@click.option("--use-reserve", is_flag=True, help="Use reserve tokens instead of holdout tokens")
def pumpfun_backtest(
    model_dir: str,
    minutes: int,
    test_window: int,
    target_mode: str,
    max_tokens: int | None,
    max_samples: int | None,
    use_reserve: bool,
) -> None:
    """Backtest pump.fun model on holdout or reserve tokens."""
    result = pumpfun_backtest_command(
        model_dir=model_dir,
        minutes=minutes,
        test_window=test_window,
        target_mode=target_mode,
        max_tokens=max_tokens,
        max_samples=max_samples,
        use_reserve_tokens=use_reserve,
    )
    click.echo(f"Backtest metrics: {result}")


@cli.command("pumpfun-predict")
@click.option("--token-id", required=True, help="Token ID to predict")
@click.option("--model-dir", required=True, help="Directory to load model")
@click.option("--minutes", type=int, default=10)
@click.option("--target-mode", type=click.Choice(["sum", "direct"]), default="sum")
def pumpfun_predict(token_id: str, model_dir: str, minutes: int, target_mode: str) -> None:
    """Predict pump.fun token price movement."""
    result = pumpfun_predict_command(
        token_id=token_id, model_dir=model_dir, minutes=minutes, target_mode=target_mode
    )
    click.echo(f"Pump.fun prediction: {result}")


@cli.command("pumpfun-classify-train")
@click.option("--model-dir", default="models_pumpfun_classifier_v9", help="Directory to save classifier")
@click.option("--horizon-minutes", type=int, default=10)
@click.option("--hidden-dim", type=int, default=128)
@click.option("--dropout", type=float, default=0.1)
@click.option("--num-layers", type=int, default=2)
@click.option("--epochs", type=int, default=20)
@click.option("--batch-size", type=int, default=512)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--label-threshold", type=float, default=0.0)
@click.option("--holdout-count", type=int, default=12)
@click.option("--min-token-samples", type=int, default=0)
@click.option("--no-pos-weight", is_flag=True, help="Disable positive class weighting")
@click.option("--no-normalize", is_flag=True, help="Disable feature normalization")
@click.option("--max-samples", type=int, default=10_000_000, help="Maximum samples to load (default: 10M to prevent OOM, set 0 for unlimited)")
def pumpfun_classify_train(
    model_dir: str,
    horizon_minutes: int,
    hidden_dim: int,
    dropout: float,
    num_layers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    label_threshold: float,
    holdout_count: int,
    min_token_samples: int,
    no_pos_weight: bool,
    no_normalize: bool,
    max_samples: int,
) -> None:
    """Train pump.fun direction classifier."""
    # Convert 0 to None for unlimited
    max_samples_param = None if max_samples == 0 else max_samples
    
    result = pumpfun_classify_train_command(
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
        use_pos_weight=not no_pos_weight,
        normalize_features=not no_normalize,
        max_samples=max_samples_param,
    )
    click.echo(f"Pump.fun classifier training complete. Metrics: {result['metrics']}")
    click.echo(f"Holdout tokens saved: {len(result['holdout_tokens'])}")


@cli.command("pumpfun-classify-backtest")
@click.option("--model-dir", default="models_pumpfun_classifier_v9", help="Directory to load classifier")
@click.option("--max-tokens", type=int, default=None)
@click.option("--max-samples", type=int, default=None)
def pumpfun_classify_backtest(
    model_dir: str, max_tokens: int | None, max_samples: int | None
) -> None:
    """Backtest pump.fun direction classifier."""
    result = pumpfun_classify_backtest_command(
        model_dir=model_dir, max_tokens=max_tokens, max_samples=max_samples
    )
    click.echo(f"Classifier backtest metrics: {result}")


@cli.command("pumpfun-clear-processed")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def pumpfun_clear_processed(force: bool) -> None:
    """Clear all processed candle and feature data (keeps original trades/tokens).
    
    SAFE: This only clears derived data (candles/features), NOT the original pump.fun trade data.
    After clearing, run 'pumpfun-sync' to regenerate candles/features from trades.
    """
    if not force:
        click.confirm(
            "This will delete ALL processed candle and feature data (but keep original trades/tokens). Continue?",
            abort=True
        )
    
    result = pumpfun_clear_processed_command()
    if result['candles_deleted'] == -1:
        click.echo("✓ Cleared all processed data from pump_candles_1m and pump_features_1m tables")
    else:
        click.echo(
            f"✓ Cleared processed data: {result['candles_deleted']} candles, {result['features_deleted']} features deleted"
        )
    click.echo("Run 'pumpfun-sync' to regenerate candles/features from trades.")


@cli.command("pumpfun-select-reserve")
@click.option("--count", type=int, default=50, help="Number of reserve tokens to select")
def pumpfun_select_reserve(count: int) -> None:
    """Select reserve tokens for evaluation (completely separate from training/holdout).
    
    Reserve tokens are tokens that meet minimum data requirements but were NOT used
    in any training (not in holdout sets). These provide unbiased evaluation data.
    """
    result = pumpfun_select_reserve_command(reserve_count=count)
    click.echo(f"✓ Selected {result['count']} reserve tokens")
    click.echo(f"Reserve tokens saved to: pumpfun_train/reserve_tokens.txt")


if __name__ == "__main__":
    cli()
