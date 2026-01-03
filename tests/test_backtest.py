"""Tests for backtest components."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.metrics_calculator import calculate_metrics, calculate_prediction_metrics
from src.backtest.signal_executor import execute_signals
from src.backtest.walk_forward import walk_forward_backtest
from src.training.config import TrainingConfig


def _sample_data(rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "return": [0.001] * rows,
            "close": [100 + i for i in range(rows)],
        }
    )


def test_execute_signals_generates_trades():
    data = _sample_data(5)
    signals = ["long", "long", "flat", "short", "flat"]
    trades = execute_signals(data, signals)
    assert len(trades) >= 1


def test_execute_signals_errors():
    data = _sample_data(3)
    try:
        execute_signals(data, ["long"])
    except ValueError:
        assert True
    else:
        assert False

    data_missing = data.drop(columns=["close"])
    try:
        execute_signals(data_missing, ["long", "flat", "flat"])
    except ValueError:
        assert True
    else:
        assert False


def test_metrics_calculator_no_trades():
    metrics = calculate_metrics([])
    assert metrics["num_trades"] == 0


def test_metrics_calculator_with_trades():
    data = _sample_data(4)
    trades = execute_signals(data, ["long", "flat", "short", "flat"])
    metrics = calculate_metrics(trades)
    assert metrics["num_trades"] >= 1


def test_prediction_metrics():
    y_true = [0.1, -0.2, 0.0, 0.3, -0.1]
    y_pred = [0.2, -0.1, 0.0, -0.4, -0.2]
    metrics = calculate_prediction_metrics(y_true, y_pred)
    assert metrics["directional_accuracy"] >= 0.0
    assert metrics["max_correct_streak"] >= 0.0


def test_prediction_metrics_empty():
    metrics = calculate_prediction_metrics([], [])
    assert metrics["directional_accuracy"] == 0.0


def test_prediction_metrics_mismatch():
    with pytest.raises(ValueError):
        calculate_prediction_metrics([0.1], [0.1, 0.2])


def test_walk_forward_backtest(tmp_path):
    data = _sample_data(40)
    config = TrainingConfig()
    result = walk_forward_backtest(
        data,
        config,
        train_size=20,
        val_size=5,
        test_size=5,
        step_size=5,
        model_dir=str(tmp_path),
        purge=1,
    )
    assert "total_return" in result.aggregated
    assert "directional_accuracy" in result.aggregated
