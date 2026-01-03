"""Tests for terminal menu."""

from __future__ import annotations

import types

from src.cli import menu as menu_module


def test_run_menu_exit(monkeypatch):
    inputs = iter(["0"])

    def fake_input(prompt: str = "") -> str:
        return next(inputs)

    monkeypatch.setattr(menu_module.console, "input", fake_input)
    monkeypatch.setattr(menu_module.console, "print", lambda *args, **kwargs: None)

    menu_module.run_menu()


def test_menu_actions(monkeypatch):
    monkeypatch.setattr(menu_module.console, "input", lambda *args, **kwargs: "")
    monkeypatch.setattr(menu_module.console, "print", lambda *args, **kwargs: None)

    dummy_commands = types.SimpleNamespace(
        train_command=lambda: {"mae": 0.1},
        predict_command=lambda *args, **kwargs: {"prediction": {}, "signal": "flat"},
        backtest_command=lambda *args, **kwargs: {"total_return": 0.0},
    )

    monkeypatch.setattr(menu_module, "run_menu", lambda: None)
    monkeypatch.setattr("src.cli.commands.train_command", dummy_commands.train_command)
    monkeypatch.setattr("src.cli.commands.predict_command", dummy_commands.predict_command)
    monkeypatch.setattr("src.cli.commands.backtest_command", dummy_commands.backtest_command)

    menu_module.show_train_menu()
    menu_module.show_predict_menu()
    menu_module.show_backtest_menu()


def test_menu_data_views(monkeypatch):
    monkeypatch.setattr(menu_module.console, "print", lambda *args, **kwargs: None)
    monkeypatch.setattr(menu_module.console, "input", lambda *args, **kwargs: "")

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyDB:
        def session(self):
            return DummySession()

    monkeypatch.setattr("src.database.connection.get_db_manager", lambda: DummyDB())
    monkeypatch.setattr("src.database.operations.count_candles", lambda *_: 0)
    monkeypatch.setattr("src.database.operations.get_all_models", lambda *_: [])

    menu_module.show_data_menu()
    menu_module.show_model_info()


def test_menu_load_data(monkeypatch):
    inputs = iter(["/tmp/kucoin.csv", "n", ""])
    monkeypatch.setattr(menu_module.console, "input", lambda *args, **kwargs: next(inputs))
    monkeypatch.setattr(menu_module.console, "print", lambda *args, **kwargs: None)

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyDB:
        def session(self):
            return DummySession()

    class DummyPipeline:
        def __init__(self, session):
            self.session = session

        def run_full_pipeline(self, file_path, replace_existing=False):
            return {"candles": 1, "features": 1}

    monkeypatch.setattr("src.database.connection.get_db_manager", lambda: DummyDB())
    monkeypatch.setattr("src.data.pipeline.DataPipeline", DummyPipeline)

    menu_module.show_load_data_menu()
