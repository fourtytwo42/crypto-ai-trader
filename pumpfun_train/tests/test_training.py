"""Tests for training infrastructure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pumpfun_train.config import TrainingConfig
from pumpfun_train.data_preparation import split_by_ratio, to_neuralforecast_format, walk_forward_splits
from pumpfun_train.evaluator import calculate_metrics
from pumpfun_train.model_factory import SimpleQuantileModel, create_model
from pumpfun_train.trainer import train_model


def _sample_df(rows: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
            "return": np.linspace(-0.01, 0.02, rows),
        }
    )


def test_training_config_validation():
    config = TrainingConfig()
    config.validate()

    with pytest.raises(ValueError):
        TrainingConfig(horizon=0).validate()


def test_split_by_ratio():
    df = _sample_df(20)
    splits = split_by_ratio(df, train_ratio=0.5, val_ratio=0.25)
    assert len(splits.train) == 10
    assert len(splits.val) == 5
    assert len(splits.test) == 5


def test_to_neuralforecast_format():
    df = _sample_df(5)
    nf = to_neuralforecast_format(df)
    assert list(nf.columns) == ["unique_id", "ds", "y"]


def test_to_neuralforecast_format_with_features():
    df = _sample_df(5)
    df["feat"] = np.arange(5)
    nf = to_neuralforecast_format(df, feature_cols=["feat"])
    assert "feat" in nf.columns


def test_to_neuralforecast_format_missing_feature():
    df = _sample_df(5)
    with pytest.raises(ValueError):
        to_neuralforecast_format(df, feature_cols=["missing"])


def test_walk_forward_splits():
    df = _sample_df(25)
    splits = walk_forward_splits(df, train_size=10, val_size=5, test_size=5, step_size=5, purge=1)
    assert len(splits) == 1
    assert len(splits[0].train) == 10


def test_simple_model_fit_predict():
    model = SimpleQuantileModel(quantiles=[0.1, 0.5, 0.9])
    data = np.array([0.0, 1.0, 2.0])
    model.fit(data)
    preds = model.predict(2)
    assert set(preds.keys()) == {"q10", "q50", "q90"}


def test_create_model_force_simple():
    config = TrainingConfig()
    model = create_model(config, force_simple=True)
    assert isinstance(model, SimpleQuantileModel)


def test_calculate_metrics():
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([0.0, 1.0, 1.5])
    metrics = calculate_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert metrics["rmse"] >= 0


def test_train_model_saves_artifacts(tmp_path: Path):
    df = _sample_df(20)
    config = TrainingConfig()
    result = train_model(
        config,
        df.iloc[:15],
        df.iloc[15:],
        model_dir=tmp_path,
        target_col="return",
        force_simple=True,
    )
    assert result.model_path.exists()
    assert result.metadata_path.exists()


def test_train_model_missing_target(tmp_path: Path):
    df = _sample_df(10)
    config = TrainingConfig()
    with pytest.raises(ValueError):
        train_model(
            config,
            df.drop(columns=["return"]),
            df.iloc[:2],
            model_dir=tmp_path,
            target_col="return",
            force_simple=True,
        )


def test_create_model_fallback(monkeypatch):
    config = TrainingConfig()
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "neuralforecast":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    model = create_model(config, force_simple=False)
    assert isinstance(model, SimpleQuantileModel)
