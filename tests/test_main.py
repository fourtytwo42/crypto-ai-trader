"""Tests for main entry point."""

from __future__ import annotations

import src.main as main_module


def test_main_success(monkeypatch):
    monkeypatch.setattr("src.cli.cli.cli", lambda: None)
    assert main_module.main() == 0


def test_main_keyboard_interrupt(monkeypatch):
    def raise_interrupt():
        raise KeyboardInterrupt()

    monkeypatch.setattr("src.cli.cli.cli", raise_interrupt)
    assert main_module.main() == 130


def test_main_error(monkeypatch):
    def raise_error():
        raise RuntimeError("boom")

    monkeypatch.setattr("src.cli.cli.cli", raise_error)
    assert main_module.main() == 1
