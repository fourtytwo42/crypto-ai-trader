"""CLI command interface for Bitcoin Trading Model.

Provides Click-based command-line interface for all operations.
"""

import click
import structlog

from src.cli.commands import (
    backtest_command,
    load_data_command,
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
@click.option("--replace", is_flag=True, help="Replace existing data")
def load_data(file_path: str, replace: bool) -> None:
    """Load CSV data into database.

    FILE_PATH: Path to Kraken CSV file.
    """
    logger.info("Loading data", file_path=file_path, replace=replace)

    try:
        result = load_data_command(file_path, replace_existing=replace)
        click.echo(f"Loaded {result['candles']} candles and {result['features']} features")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


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
def backtest(model_dir: str) -> None:
    """Run backtest on historical data.

    Evaluates model performance using walk-forward testing.
    """
    result = backtest_command(model_dir)
    click.echo(f"Backtest metrics: {result}")


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
