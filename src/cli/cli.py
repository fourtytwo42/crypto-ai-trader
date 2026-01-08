"""CLI command interface for Bitcoin Trading Model.

Provides Click-based command-line interface for all operations.
"""

import click
import structlog

from src.cli.commands import (
    backtest_command,
    forecast_backtest_command,
    forecast_holdout_24h_command,
    forecast_train_command,
    forecast_predict_command,
    load_data_command,
    load_hourly_data_command,
    menu_command,
    predict_command,
    quick_predict_command,
    train_command,
    pumpfun_backtest_command,
    pumpfun_classify_backtest_command,
    pumpfun_classify_train_command,
    pumpfun_predict_command,
    pumpfun_sync_command,
    pumpfun_train_command,
)

logger = structlog.get_logger(__name__)


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_mlp_units(value: str | None) -> list[list[int]] | None:
    if not value:
        return None
    stacks = [stack.strip() for stack in value.split(";") if stack.strip()]
    units: list[list[int]] = []
    for stack in stacks:
        sep = "|" if "|" in stack else ","
        inner = [int(item.strip()) for item in stack.split(sep) if item.strip()]
        if not inner:
            continue
        units.append(inner)
    return units or None


@click.group(invoke_without_command=True)
@click.option("--menu", is_flag=True, help="Launch interactive menu")
@click.pass_context
def cli(ctx: click.Context, menu: bool) -> None:
    """Bitcoin Trading Model CLI.

    Use --menu for interactive mode or run specific commands.
    """
    if ctx.invoked_subcommand is None:
        if menu:
            menu_command()
        else:
            click.echo(ctx.get_help())


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--symbol", default="BTC-USDT", help="Trading pair symbol (e.g., BTC-USDT)")
@click.option("--replace", is_flag=True, help="Replace existing data")
def load_data(file_path: str, symbol: str, replace: bool) -> None:
    """Load CSV data into database.

    FILE_PATH: Path to Kraken CSV file.
    """
    logger.info("Loading data", file_path=file_path, replace=replace)

    try:
        result = load_data_command(file_path, symbol=symbol, replace_existing=replace)
        click.echo(f"Loaded {result['candles']} candles and {result['features']} features")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option(
    "--symbols",
    default="BTC-USDT,ETH-USDT,LTC-USDT",
    help="Comma-separated KuCoin symbols to backfill.",
)
@click.option("--years-back", type=float, default=8.0, help="Years of hourly data to pull")
@click.option("--hours-back", type=int, default=None, help="Override hours back")
@click.option("--replace", is_flag=True, help="Replace existing data")
def load_hourly(symbols: str, years_back: float, hours_back: int | None, replace: bool) -> None:
    """Load hourly KuCoin data for one or more symbols."""
    hours = hours_back if hours_back is not None else int(years_back * 365 * 24)
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        raise SystemExit("No symbols provided.")
    for symbol in symbol_list:
        result = load_hourly_data_command(
            symbol=symbol,
            hours_back=hours,
            replace_existing=replace,
        )
        click.echo(f"{symbol}: loaded {result} candles")


@cli.command()
def train() -> None:
    """Train a new model.

    Trains model using data from database.
    """
    metrics = train_command()
    click.echo(f"Training complete. Metrics: {metrics}")


@cli.command()
@click.option("--model-dir", type=click.Path(exists=True), required=True)
def predict(model_dir: str) -> None:
    """Generate prediction using trained model.

    Uses latest data to generate prediction.
    """
    result = predict_command(model_dir)
    click.echo(f"Prediction: {result}")


@cli.command()
@click.option("--model-dir", type=click.Path(exists=True), required=True)
@click.option("--start-window", type=int, default=None, help="Start at a specific window (1-based)")
@click.option("--max-windows", type=int, default=None, help="Maximum number of windows to run")
def backtest(model_dir: str, start_window: int | None, max_windows: int | None) -> None:
    """Run backtest on historical data.

    Evaluates model performance using walk-forward testing.
    """
    result = backtest_command(
        model_dir,
        start_window=start_window,
        max_windows=max_windows,
    )
    click.echo(f"Backtest metrics: {result}")


@cli.command()
@click.option("--horizon", default=12, type=int, help="Forecast horizon in hours")
@click.option("--start-window", type=int, default=None, help="Start at a specific window (1-based)")
@click.option("--max-windows", type=int, default=None, help="Maximum number of windows to run")
@click.option("--train-size", type=int, default=None, help="Training window size in rows")
@click.option("--val-size", type=int, default=None, help="Validation window size in rows")
@click.option("--test-size", type=int, default=None, help="Test window size in rows")
@click.option("--step-size", type=int, default=None, help="Step size in rows")
@click.option("--model-type", default="nhits", type=click.Choice(["patchtst", "nhits"]))
@click.option("--context-length", type=int, default=None, help="Input context length in rows")
@click.option("--hidden-size", type=int, default=None, help="Model hidden size")
@click.option("--num-layers", type=int, default=None, help="Number of model layers")
@click.option("--patch-length", type=int, default=None, help="Patch length (PatchTST)")
@click.option("--stride", type=int, default=None, help="Patch stride (PatchTST)")
@click.option(
    "--horizon-weight-mode",
    default="none",
    type=click.Choice(["none", "linear", "quadratic", "cubic", "exp"]),
    help="Weight later horizons more in the loss",
)
@click.option(
    "--loss-type",
    default="mae",
    type=click.Choice(["mae", "huber"]),
    help="Loss function for training",
)
@click.option(
    "--close-target",
    default="log_close",
    type=click.Choice(["log_close", "return"]),
    help="Target for close forecasting (predict log_close or log return).",
)
@click.option("--epochs", type=int, default=None, help="Training epochs per window")
@click.option("--batch-size", type=int, default=None, help="Training batch size")
@click.option("--learning-rate", type=float, default=None, help="Learning rate")
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated symbols for multi-asset training (e.g., BTC-USDT,ETH-USDT).",
)
@click.option("--multi-asset", is_flag=True, help="Enable multi-asset training.")
def forecast_backtest(
    horizon: int,
    start_window: int | None,
    max_windows: int | None,
    train_size: int | None,
    val_size: int | None,
    test_size: int | None,
    step_size: int | None,
    model_type: str,
    context_length: int | None,
    hidden_size: int | None,
    num_layers: int | None,
    patch_length: int | None,
    stride: int | None,
    horizon_weight_mode: str,
    loss_type: str,
    close_target: str,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    symbols: str | None,
    multi_asset: bool,
) -> None:
    """Run forecast backtest for next-hour candles and volume."""
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    result = forecast_backtest_command(
        horizon=horizon,
        start_window=start_window,
        max_windows=max_windows,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=step_size,
        model_type=model_type,
        context_length=context_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        horizon_weight_mode=horizon_weight_mode,
        loss_type=loss_type,
        close_target=close_target,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        symbols=symbol_list,
        multi_asset=multi_asset,
    )
    click.echo(f"Forecast backtest metrics: {result}")


@cli.command()
@click.option("--model-dir", type=click.Path(), required=True)
@click.option("--horizon", default=12, type=int, help="Forecast horizon in hours")
@click.option("--context-length", type=int, default=None, help="Input context length in rows")
@click.option("--hidden-size", type=int, default=None, help="Model hidden size")
@click.option("--num-layers", type=int, default=None, help="Number of model layers")
@click.option("--patch-length", type=int, default=None, help="Patch length (PatchTST)")
@click.option("--stride", type=int, default=None, help="Patch stride (PatchTST)")
@click.option(
    "--horizon-weight-mode",
    default="none",
    type=click.Choice(["none", "linear", "quadratic", "cubic", "exp"]),
    help="Weight later horizons more in the loss",
)
@click.option(
    "--loss-type",
    default="mae",
    type=click.Choice(["mae", "huber"]),
    help="Loss function for training",
)
@click.option(
    "--close-target",
    default="log_close",
    type=click.Choice(["log_close", "return"]),
    help="Target for close forecasting (predict log_close or log return).",
)
@click.option("--epochs", type=int, default=None, help="Training epochs")
@click.option("--batch-size", type=int, default=None, help="Training batch size")
@click.option("--learning-rate", type=float, default=None, help="Learning rate")
@click.option("--model-type", default="nhits", type=click.Choice(["patchtst", "nhits"]))
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated symbols for multi-asset training (e.g., BTC-USDT,ETH-USDT).",
)
@click.option("--multi-asset", is_flag=True, help="Enable multi-asset training.")
def forecast_train(
    model_dir: str,
    horizon: int,
    context_length: int | None,
    hidden_size: int | None,
    num_layers: int | None,
    patch_length: int | None,
    stride: int | None,
    horizon_weight_mode: str,
    loss_type: str,
    close_target: str,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    model_type: str,
    symbols: str | None,
    multi_asset: bool,
) -> None:
    """Train and save forecast models for close and volume."""
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    result = forecast_train_command(
        model_dir=model_dir,
        horizon=horizon,
        context_length=context_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        horizon_weight_mode=horizon_weight_mode,
        loss_type=loss_type,
        close_target=close_target,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_type=model_type,
        symbols=symbol_list,
        multi_asset=multi_asset,
    )
    click.echo(f"Forecast training metrics: {result}")


@cli.command()
@click.option("--model-dir", type=click.Path(), required=True)
@click.option("--horizon", default=12, type=int, help="Forecast horizon in hours")
def forecast_predict(model_dir: str, horizon: int) -> None:
    """Generate next-horizon forecasts using saved models."""
    result = forecast_predict_command(model_dir=model_dir, horizon=horizon)
    click.echo(f"Forecast prediction: {result}")


@cli.command("forecast-holdout-24h")
@click.option("--context-length", type=int, required=True, help="Input context length in rows")
@click.option("--hidden-size", type=int, required=True, help="Model hidden size")
@click.option("--num-layers", type=int, required=True, help="Number of model layers")
@click.option("--patch-length", type=int, required=True, help="Patch length (PatchTST)")
@click.option("--stride", type=int, required=True, help="Patch stride (PatchTST)")
@click.option("--loss-type", default="huber", type=click.Choice(["mae", "huber"]))
@click.option("--epochs", type=int, required=True, help="Training epochs")
@click.option("--batch-size", type=int, required=True, help="Training batch size")
@click.option("--learning-rate", type=float, required=True, help="Learning rate")
@click.option("--model-type", default="patchtst", type=click.Choice(["patchtst", "nhits"]))
@click.option(
    "--nhits-stack-types",
    default=None,
    help="NHITS stack types (comma-separated, e.g., identity,identity,identity).",
)
@click.option(
    "--nhits-n-blocks",
    default=None,
    help="NHITS blocks per stack (comma-separated, e.g., 1,1,1).",
)
@click.option(
    "--nhits-mlp-units",
    default=None,
    help="NHITS MLP units per stack (e.g., 512|512;512|512;512|512).",
)
@click.option(
    "--nhits-n-pool-kernel-size",
    default=None,
    help="NHITS pool kernel sizes (comma-separated, e.g., 2,2,1).",
)
@click.option(
    "--nhits-n-freq-downsample",
    default=None,
    help="NHITS freq downsample factors (comma-separated, e.g., 4,2,1).",
)
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated symbols for multi-asset training (e.g., BTC-USDT,ETH-USDT).",
)
@click.option("--multi-asset", is_flag=True, help="Enable multi-asset training.")
def forecast_holdout_24h(
    context_length: int,
    hidden_size: int,
    num_layers: int,
    patch_length: int,
    stride: int,
    loss_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_type: str,
    nhits_stack_types: str | None,
    nhits_n_blocks: str | None,
    nhits_mlp_units: str | None,
    nhits_n_pool_kernel_size: str | None,
    nhits_n_freq_downsample: str | None,
    symbols: str | None,
    multi_asset: bool,
) -> None:
    """Train on data older than last 24 hours and evaluate on latest 24 hours."""
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    result = forecast_holdout_24h_command(
        context_length=context_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        loss_type=loss_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_type=model_type,
        nhits_stack_types=_parse_csv_strings(nhits_stack_types),
        nhits_n_blocks=_parse_csv_ints(nhits_n_blocks),
        nhits_mlp_units=_parse_mlp_units(nhits_mlp_units),
        nhits_n_pool_kernel_size=_parse_csv_ints(nhits_n_pool_kernel_size),
        nhits_n_freq_downsample=_parse_csv_ints(nhits_n_freq_downsample),
        symbols=symbol_list,
        multi_asset=multi_asset,
    )
    click.echo(f"Holdout 24h metrics: {result}")


@cli.command("quick-predict")
@click.option(
    "--hours",
    type=int,
    default=24,
    help="Number of hours to predict ahead (24, 48, 72, etc.)",
)
@click.option(
    "--retrain",
    is_flag=True,
    help="Retrain the model before prediction. If not set, uses existing model.",
)
@click.option(
    "--symbols",
    default=None,
    help="Comma-separated symbols (any KuCoin pair, default BTC-USDT,ETH-USDT,LTC-USDT,XRP-USDT)",
)
@click.option(
    "--model-dir",
    default="models_nhits_best",
    help="Directory to save/load model",
)
def quick_predict(
    hours: int,
    retrain: bool,
    symbols: str | None,
    model_dir: str,
) -> None:
    """Fetch latest data, optionally retrain, and predict prices.

    This is the main prediction command that:

    \b
    1. Pulls the latest candle data from KuCoin
    2. Updates the database with new candles
    3. Optionally retrains the NHITS model (--retrain)
    4. Outputs price predictions for all symbols

    \b
    Examples:
      # Quick prediction using existing model (inference only):
      python -m src.main quick-predict --hours 24

    \b
      # Retrain model first, then predict:
      python -m src.main quick-predict --hours 24 --retrain

    \b
      # Predict 48 hours ahead:
      python -m src.main quick-predict --hours 48

    \b
      # Predict specific symbols:
      python -m src.main quick-predict --hours 24 --symbols BTC-USDT,ETH-USDT
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    try:
        result = quick_predict_command(
            hours=hours,
            retrain=retrain,
            symbols=symbol_list,
            model_dir=model_dir,
        )

        # Pretty print the results
        click.echo("\n" + "=" * 60)
        click.echo(f"PRICE PREDICTIONS ({hours}h ahead)")
        click.echo("=" * 60)

        if retrain:
            click.echo(f"Model: Retrained (saved to {model_dir})")
        else:
            click.echo(f"Model: Loaded from {model_dir}")

        click.echo(f"Generated: {result['generated_at']}")
        click.echo("-" * 60)

        predictions = result.get("predictions", {})
        for symbol, pred in predictions.items():
            direction_icon = "[UP]" if pred["direction"] == "UP" else "[DOWN]" if pred["direction"] == "DOWN" else "[FLAT]"
            click.echo(f"\n{symbol}:")
            click.echo(f"  Current Price:   ${pred['current_price']:,.2f}")
            click.echo(f"  Predicted Price: ${pred['predicted_price']:,.2f}")
            click.echo(f"  Change:          ${pred['price_change']:+,.2f} ({pred['price_change_pct']:+.2f}%)")
            click.echo(f"  Direction:       {direction_icon}")
            click.echo(f"  Last Data:       {pred['last_data_timestamp']}")

        click.echo("\n" + "=" * 60)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@cli.command("pumpfun-sync")
@click.option("--min-age-minutes", type=int, default=60)
@click.option("--active-age-minutes", type=int, default=240)
@click.option("--min-total-trades", type=int, default=10)
@click.option("--min-recent-trades", type=int, default=10)
@click.option("--recent-window-minutes", type=int, default=30)
@click.option("--max-tokens", type=int, default=None)
@click.option("--skip-price-lookup", is_flag=True, help="Skip SOL/USD lookup for missing USD fields")
@click.option("--replace", is_flag=True, help="Replace existing candles/features")
def pumpfun_sync(
    min_age_minutes: int,
    active_age_minutes: int,
    min_total_trades: int,
    min_recent_trades: int,
    recent_window_minutes: int,
    max_tokens: int | None,
    skip_price_lookup: bool,
    replace: bool,
) -> None:
    """Sync pump.fun trades into minute candles/features."""
    result = pumpfun_sync_command(
        min_age_minutes=min_age_minutes,
        active_age_minutes=active_age_minutes,
        min_total_trades=min_total_trades,
        min_recent_trades=min_recent_trades,
        recent_window_minutes=recent_window_minutes,
        replace_existing=replace,
        max_tokens=max_tokens,
        price_lookup_enabled=not skip_price_lookup,
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
@click.option("--model-dir", default="models_pumpfun_sum10_ctx240_lr1e4_e30", help="Directory to load model")
@click.option("--minutes", type=int, default=10)
@click.option("--test-window", type=int, default=240)
@click.option("--target-mode", type=click.Choice(["sum", "direct"]), default="sum")
@click.option("--max-tokens", type=int, default=None)
@click.option("--max-samples", type=int, default=None)
def pumpfun_backtest(
    model_dir: str,
    minutes: int,
    test_window: int,
    target_mode: str,
    max_tokens: int | None,
    max_samples: int | None,
) -> None:
    """Backtest pump.fun model on holdout tokens."""
    result = pumpfun_backtest_command(
        model_dir=model_dir,
        minutes=minutes,
        test_window=test_window,
        target_mode=target_mode,
        max_tokens=max_tokens,
        max_samples=max_samples,
    )
    click.echo(f"Backtest metrics: {result}")


@cli.command("pumpfun-predict")
@click.option("--token-id", required=True, help="Token ID to predict")
@click.option("--model-dir", default="models_pumpfun_sum10_ctx240_lr1e4_e30", help="Directory to load model")
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
) -> None:
    """Train pump.fun direction classifier."""
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


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, type=int, help="API port")
def api(host: str, port: int) -> None:
    """Start the FastAPI server.

    Launches REST API for programmatic access.
    """
    import uvicorn
    from src.api.main import app

    click.echo(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
