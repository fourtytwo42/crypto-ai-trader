"""Interactive terminal menu for Bitcoin Trading Model.

Provides Rich-based interactive menu for all operations.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def run_menu() -> None:
    """Display main menu and handle selection."""
    while True:
        console.clear()
        console.print(Panel.fit("[bold blue]Bitcoin Trading Model[/bold blue]", padding=(1, 4)))

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Option", style="cyan")
        table.add_column("Description")

        table.add_row("1", "Train Model")
        table.add_row("2", "Make Prediction")
        table.add_row("3", "Run Backtest")
        table.add_row("4", "View Data")
        table.add_row("5", "Model Info")
        table.add_row("6", "Load Data")
        table.add_row("0", "Exit")

        console.print(table)
        console.print()

        choice = console.input("[bold]Select option:[/bold] ")

        if choice == "0":
            console.print("[yellow]Goodbye![/yellow]")
            break
        elif choice == "1":
            show_train_menu()
        elif choice == "2":
            show_predict_menu()
        elif choice == "3":
            show_backtest_menu()
        elif choice == "4":
            show_data_menu()
        elif choice == "5":
            show_model_info()
        elif choice == "6":
            show_load_data_menu()
        else:
            console.print("[red]Invalid option[/red]")
            console.input("Press Enter to continue...")


def show_main_menu() -> None:
    """Display main menu for one interaction."""
    run_menu()


def show_train_menu() -> None:
    """Display training menu."""
    console.print(Panel("[bold]Train Model[/bold]"))
    from src.cli.commands import train_command

    try:
        metrics = train_command()
        console.print(f"[green]Training complete: {metrics}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    console.input("Press Enter to continue...")


def show_predict_menu() -> None:
    """Display prediction menu."""
    console.print(Panel("[bold]Make Prediction[/bold]"))
    from src.cli.commands import predict_command

    model_dir = console.input("Enter model directory: ")
    if not model_dir:
        console.print("[yellow]Cancelled[/yellow]")
        console.input("Press Enter to continue...")
        return

    try:
        result = predict_command(model_dir)
        console.print(f"[green]Prediction: {result}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    console.input("Press Enter to continue...")


def show_backtest_menu() -> None:
    """Display backtest menu."""
    console.print(Panel("[bold]Run Backtest[/bold]"))
    from src.cli.commands import backtest_command

    model_dir = console.input("Enter model directory: ")
    if not model_dir:
        console.print("[yellow]Cancelled[/yellow]")
        console.input("Press Enter to continue...")
        return

    try:
        metrics = backtest_command(model_dir)
        console.print(f"[green]Backtest metrics: {metrics}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    console.input("Press Enter to continue...")


def show_data_menu() -> None:
    """Display data viewing menu."""
    console.print(Panel("[bold]View Data[/bold]"))

    from src.database.connection import get_db_manager
    from src.database.operations import count_candles

    try:
        db = get_db_manager()
        with db.session() as session:
            candle_count = count_candles(session)
        console.print(f"Total candles in database: {candle_count}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    console.input("Press Enter to continue...")


def show_model_info() -> None:
    """Display model information."""
    console.print(Panel("[bold]Model Info[/bold]"))

    from src.database.connection import get_db_manager
    from src.database.operations import get_all_models

    try:
        db = get_db_manager()
        with db.session() as session:
            models = get_all_models(session)

        if not models:
            console.print("No models found in database")
        else:
            table = Table(title="Available Models")
            table.add_column("ID")
            table.add_column("Name")
            table.add_column("Type")
            table.add_column("Version")

            for model in models:
                table.add_row(
                    str(model.id),
                    model.name,
                    model.model_type,
                    model.version,
                )
            console.print(table)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    console.input("Press Enter to continue...")


def show_load_data_menu() -> None:
    """Display data loading menu."""
    console.print(Panel("[bold]Load Data[/bold]"))

    file_path = console.input("Enter CSV file path: ")
    if not file_path:
        console.print("[yellow]Cancelled[/yellow]")
        console.input("Press Enter to continue...")
        return

    replace = console.input("Replace existing data? [y/N]: ").lower() == "y"

    from src.database.connection import get_db_manager
    from src.data.pipeline import DataPipeline

    try:
        db = get_db_manager()
        with db.session() as session:
            pipeline = DataPipeline(session)
            result = pipeline.run_full_pipeline(file_path, replace_existing=replace)

        console.print(f"[green]Loaded {result['candles']} candles and {result['features']} features[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    console.input("Press Enter to continue...")
