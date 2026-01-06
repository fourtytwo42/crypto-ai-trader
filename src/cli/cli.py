"""CLI command interface for Bitcoin Trading Model.

Provides Click-based command-line interface for all operations.
"""

import click
import structlog

from src.cli.commands import (
    backtest_command,
    forecast_backtest_command,
    forecast_train_command,
    forecast_predict_command,
    load_data_command,
    load_hourly_data_command,
    menu_command,
    predict_command,
    train_command,
)

logger = structlog.get_logger(__name__)


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
) -> None:
    """Train and save forecast models for close and volume."""
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
    )
    click.echo(f"Forecast training metrics: {result}")


@cli.command()
@click.option("--model-dir", type=click.Path(), required=True)
@click.option("--horizon", default=12, type=int, help="Forecast horizon in hours")
def forecast_predict(model_dir: str, horizon: int) -> None:
    """Generate next-horizon forecasts using saved models."""
    result = forecast_predict_command(model_dir=model_dir, horizon=horizon)
    click.echo(f"Forecast prediction: {result}")


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
