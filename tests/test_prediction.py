"""Tests for prediction components."""

from __future__ import annotations

import pandas as pd
import pytest

from src.prediction.model_loader import load_model_artifacts
from src.prediction.predictor import extract_quantiles, generate_predictions
from src.prediction.signal_generator import apply_signals, generate_signal
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
