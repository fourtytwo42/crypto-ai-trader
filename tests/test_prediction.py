"""Tests for prediction components."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.model_loader import load_model_artifacts
from src.prediction.predictor import extract_quantiles, generate_predictions
from src.prediction.signal_generator import (
    SignalConfig,
    apply_signals,
    calculate_conviction,
    calculate_effective_threshold,
    calculate_uncertainty_spread,
    generate_signal,
    generate_signal_with_position_size,
)
from src.training.config import TrainingConfig
from src.training.model_factory import SimpleQuantileModel
from src.training.trainer import save_model_artifacts


def test_generate_signal_logic():
    assert generate_signal({"q10": -0.001, "q50": 0.006, "q90": 0.01}) == "long"
    assert generate_signal({"q10": -0.01, "q50": -0.006, "q90": 0.001}) == "short"
    assert generate_signal({"q10": -0.01, "q50": 0.0, "q90": 0.01}) == "flat"


def test_apply_signals():
    preds = [
        {"q10": -0.001, "q50": 0.006, "q90": 0.01},
        {"q10": -0.01, "q50": -0.006, "q90": 0.001},
    ]
    signals = apply_signals(preds)
    assert signals == ["long", "short"]


def test_generate_predictions_simple_model():
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "return": [0.001, 0.002, -0.001],
        }
    )
    preds = generate_predictions(model, data)
    assert len(preds) == 3
    assert "q50" in preds[0]


def test_extract_quantiles():
    preds = [
        {"timestamp": "t1", "q10": 0.1, "q50": 0.2, "q90": 0.3},
        {"timestamp": "t2", "q10": 0.2, "q50": 0.3, "q90": 0.4},
    ]
    arr = extract_quantiles(preds)
    assert arr.shape == (2,)


def test_model_loader(tmp_path):
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    config = TrainingConfig()
    save_model_artifacts(model, config, {"mae": 0.1}, tmp_path)

    import pickle

    scaler_path = tmp_path / "scaler.pkl"
    with scaler_path.open("wb") as handle:
        pickle.dump({"scale": 1.0}, handle)

    bundle = load_model_artifacts(tmp_path)
    assert bundle.model is not None
    assert "config" in bundle.metadata


def test_generate_predictions_missing_timestamp():
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    data = pd.DataFrame({"return": [0.1]})
    with pytest.raises(ValueError):
        generate_predictions(model, data)


def test_generate_predictions_empty_data():
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    data = pd.DataFrame({"timestamp": pd.to_datetime([]), "return": []})
    preds = generate_predictions(model, data)
    assert preds == []


def test_model_loader_without_torch(tmp_path, monkeypatch):
    import builtins
    import pickle

    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    model.fit([0.0, 0.1, -0.1])
    model_path = tmp_path / "model.pt"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)
    (tmp_path / "metadata.json").write_text("{\"config\": {}, \"metrics\": {}}")

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    bundle = load_model_artifacts(tmp_path)
    assert bundle.model is not None


def test_generate_predictions_non_quantile_forecast():
    class DummyModel:
        def predict(self, df):
            return pd.DataFrame(
                {
                    "ds": df["ds"],
                    "DummyModel": [0.1] * len(df),
                }
            )

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "return": [0.001, 0.002, -0.001],
        }
    )
    preds = generate_predictions(DummyModel(), data)
    assert len(preds) == 3
    assert preds[0]["q10"] == preds[0]["q50"] == preds[0]["q90"]


def test_generate_predictions_short_forecast_alignment():
    class DummyModel:
        def predict(self, df):
            return pd.DataFrame(
                {
                    "ds": df["ds"].iloc[-2:].reset_index(drop=True),
                    "q10": [0.1, 0.2],
                    "q50": [0.1, 0.2],
                    "q90": [0.1, 0.2],
                }
            )

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "return": [0.001, 0.002, -0.001],
        }
    )
    preds = generate_predictions(DummyModel(), data)
    assert len(preds) == 2


def test_generate_predictions_custom_target():
    class DummyModel:
        def predict(self, df):
            return pd.DataFrame(
                {
                    "ds": df["ds"],
                    "DummyModel": [0.05] * len(df),
                }
            )

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "target_return": [0.1, -0.2],
        }
    )
    preds = generate_predictions(DummyModel(), data, target_col="target_return")
    assert len(preds) == 2


# Tests for enhanced signal generation


def test_effective_threshold_scaling():
    """Test volatility-adaptive threshold calculation."""
    # Normal volatility - no scaling
    result = calculate_effective_threshold(0.005, 0.02, 0.02)
    assert result == pytest.approx(0.005)

    # Double volatility - 2x threshold
    result = calculate_effective_threshold(0.005, 0.04, 0.02)
    assert result == pytest.approx(0.01)

    # Half volatility - 0.5x threshold
    result = calculate_effective_threshold(0.005, 0.01, 0.02)
    assert result == pytest.approx(0.0025)


def test_effective_threshold_clamping():
    """Test that threshold multiplier is clamped."""
    # Very high volatility - should be clamped to max (2.0)
    result = calculate_effective_threshold(0.005, 0.10, 0.02, multiplier_max=2.0)
    assert result == pytest.approx(0.01)

    # Very low volatility - should be clamped to min (0.5)
    result = calculate_effective_threshold(0.005, 0.001, 0.02, multiplier_min=0.5)
    assert result == pytest.approx(0.0025)


def test_effective_threshold_zero_baseline():
    """Test that zero baseline returns base threshold."""
    result = calculate_effective_threshold(0.005, 0.02, 0.0)
    assert result == 0.005


def test_uncertainty_spread():
    """Test uncertainty spread calculation."""
    pred = {"q10": -0.01, "q50": 0.005, "q90": 0.02}
    spread = calculate_uncertainty_spread(pred)
    assert spread == pytest.approx(0.03)


def test_conviction_high():
    """Test high conviction when q50 is large relative to spread."""
    pred = {"q10": 0.008, "q50": 0.01, "q90": 0.012}  # spread = 0.004
    conviction = calculate_conviction(pred)
    assert conviction > 0.9  # High conviction


def test_conviction_low():
    """Test low conviction when q50 is small relative to spread."""
    pred = {"q10": -0.02, "q50": 0.001, "q90": 0.02}  # spread = 0.04
    conviction = calculate_conviction(pred)
    assert conviction < 0.1  # Low conviction


def test_conviction_zero_spread():
    """Test conviction with zero spread returns 1.0."""
    pred = {"q10": 0.01, "q50": 0.01, "q90": 0.01}
    conviction = calculate_conviction(pred)
    assert conviction == 1.0


def test_signal_with_uncertainty_filter():
    """Test that wide uncertainty spread results in flat signal."""
    config = SignalConfig(
        base_threshold=0.005,
        uncertainty_enabled=True,
        max_uncertainty_spread=0.02,
        volatility_adaptive=False,
    )

    # Wide spread (0.05) should return flat
    pred = {"q10": -0.02, "q50": 0.01, "q90": 0.03}
    signal = generate_signal(pred, config=config)
    assert signal == "flat"

    # Narrow spread (0.01) should allow signal
    pred = {"q10": 0.005, "q50": 0.01, "q90": 0.015}
    signal = generate_signal(pred, config=config)
    assert signal == "long"


def test_signal_with_volatility_adaptive():
    """Test volatility-adaptive threshold adjustment."""
    config = SignalConfig(
        base_threshold=0.005,
        volatility_adaptive=True,
        baseline_volatility=0.02,
        uncertainty_enabled=False,
    )

    pred = {"q10": 0.003, "q50": 0.006, "q90": 0.009}

    # Low volatility - lower threshold, should trigger long
    signal = generate_signal(pred, current_volatility=0.01, config=config)
    assert signal == "long"

    # High volatility - higher threshold, same prediction goes flat
    signal = generate_signal(pred, current_volatility=0.04, config=config)
    assert signal == "flat"


def test_signal_without_volatility_data():
    """Test that signal works without current volatility."""
    config = SignalConfig(
        base_threshold=0.005,
        volatility_adaptive=True,  # Enabled but no volatility provided
        uncertainty_enabled=False,
    )

    pred = {"q10": 0.003, "q50": 0.006, "q90": 0.009}
    signal = generate_signal(pred, current_volatility=None, config=config)
    assert signal == "long"  # Falls back to base threshold


def test_signal_with_position_size():
    """Test signal generation with position sizing."""
    config = SignalConfig(
        base_threshold=0.005,
        volatility_adaptive=False,
        uncertainty_enabled=False,
    )

    # High conviction prediction
    pred = {"q10": 0.008, "q50": 0.01, "q90": 0.012}
    signal, position_size = generate_signal_with_position_size(
        pred, config=config, min_size=0.1, max_size=1.0
    )
    assert signal == "long"
    assert position_size > 0.5  # High conviction = large position

    # Flat signal = zero position
    pred = {"q10": -0.01, "q50": 0.001, "q90": 0.01}
    signal, position_size = generate_signal_with_position_size(
        pred, config=config, min_size=0.1, max_size=1.0
    )
    assert signal == "flat"
    assert position_size == 0.0


def test_apply_signals_with_volatility_series():
    """Test applying signals with volatility series."""
    config = SignalConfig(
        base_threshold=0.005,
        volatility_adaptive=True,
        baseline_volatility=0.02,
        uncertainty_enabled=False,
    )

    predictions = [
        {"q10": 0.003, "q50": 0.006, "q90": 0.009},
        {"q10": 0.003, "q50": 0.006, "q90": 0.009},
    ]
    volatility_series = [0.01, 0.04]  # Low vol, high vol

    signals = apply_signals(
        predictions,
        volatility_series=volatility_series,
        config=config,
    )

    assert signals[0] == "long"  # Low vol = lower threshold
    assert signals[1] == "flat"  # High vol = higher threshold
