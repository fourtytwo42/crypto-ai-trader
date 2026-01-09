import pandas as pd

from pumpfun_train.feature_extractor import extract_features_minute
from pumpfun_train.pipeline import build_minute_candles, _normalize_trades


def test_build_minute_candles_basic():
    trades = pd.DataFrame(
        {
            "timestamp": [100_000, 120_000, 160_000, 220_000],
            "price_usd": [1.0, 1.2, 0.9, 1.1],
            "amount_usd": [10.0, 5.0, 8.0, 6.0],
            "amount_sol": [0.5, 0.2, 0.3, 0.25],
        }
    )
    candles = build_minute_candles(trades)
    assert not candles.empty
    assert len(candles) >= 2
    assert "open" in candles.columns
    assert candles.iloc[0].open == 1.0
    assert candles.iloc[-1].close == 1.1


def test_normalize_trades_fills_usd():
    trades = pd.DataFrame(
        {
            "timestamp": [100, 101],
            "price_sol": [0.5, 0.4],
            "price_usd": [None, None],
            "amount_sol": [1.0, 2.0],
            "amount_usd": [None, None],
        }
    )

    def price_lookup(_: int) -> float:
        return 100.0

    normalized = _normalize_trades(trades, price_lookup)
    assert normalized["price_usd"].iloc[0] == 50.0
    assert normalized["amount_usd"].iloc[1] == 200.0


def test_extract_features_minute_columns():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC"),
            "open": [1.0] * 120,
            "high": [1.1] * 120,
            "low": [0.9] * 120,
            "close": [1.0] * 120,
            "volume_usd": [10.0] * 120,
        }
    )
    features = extract_features_minute(candles)
    assert "ret_mean_15" in features.columns
    assert "ret_std_60" in features.columns
