"""Tests for normalization utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.normalizer import NormalizationError, RollingNormalizer, normalize_features


def test_fit_transform_and_single(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 11.0, 12.0, 13.0]})
    normalizer = RollingNormalizer(window=2)
    transformed = normalizer.fit_transform(df, columns=["a", "b"])
    assert transformed.shape == df.shape

    single = normalizer.transform_single({"a": 5.0, "b": 14.0})
    assert "a" in single and "b" in single

    path = tmp_path / "norm.pkl"
    normalizer.save(path)
    loaded = RollingNormalizer.load(path)
    assert loaded.is_fitted


def test_transform_single_requires_fit():
    normalizer = RollingNormalizer(window=2)
    with pytest.raises(NormalizationError):
        normalizer.transform_single({"a": 1.0})


def test_normalize_features():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    normalized = normalize_features(df, columns=["a", "b"], window=2)
    assert normalized.shape == df.shape
