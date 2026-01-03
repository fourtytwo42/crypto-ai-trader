"""Market regime detection for adaptive trading.

Detects market regimes to adjust trading strategy:
- Trending (up/down): Lower thresholds to catch directional moves
- Ranging: Higher thresholds to avoid whipsaws
- High volatility: Higher thresholds to avoid false signals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Regime = Literal["trending_up", "trending_down", "ranging", "high_volatility"]


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""

    trend_threshold: float = 0.1  # Cumulative return threshold for trend detection
    vol_percentile: float = 0.8  # Percentile threshold for high volatility
    lookback: int = 30  # Periods to analyze for regime detection


def detect_regime(
    returns: pd.Series | np.ndarray,
    config: RegimeConfig | None = None,
) -> Regime:
    """Detect current market regime from recent returns.

    Classification logic:
    1. Check if volatility is in the top percentile -> high_volatility
    2. Check if cumulative return exceeds threshold -> trending_up/trending_down
    3. Otherwise -> ranging

    Args:
        returns: Historical returns (Series or array)
        config: Regime detection configuration

    Returns:
        Current regime classification
    """
    if config is None:
        config = RegimeConfig()

    returns = np.asarray(returns)

    # Need enough data for meaningful detection
    if len(returns) < config.lookback:
        return "ranging"

    recent = returns[-config.lookback :]

    # Calculate recent and historical volatility
    recent_vol = np.std(recent)
    historical_vol = np.std(returns)

    # Check for high volatility regime
    # Compare recent vol to historical distribution
    if len(returns) > config.lookback * 2:
        # Calculate rolling volatility
        rolling_vols = []
        for i in range(config.lookback, len(returns)):
            rolling_vols.append(np.std(returns[i - config.lookback : i]))
        vol_threshold = np.percentile(rolling_vols, config.vol_percentile * 100)
        if recent_vol > vol_threshold:
            return "high_volatility"
    elif recent_vol > historical_vol * 1.5:
        # Fallback: if recent vol is 50% higher than historical
        return "high_volatility"

    # Check for trending regime
    cumulative_return = np.sum(recent)
    if cumulative_return > config.trend_threshold:
        return "trending_up"
    elif cumulative_return < -config.trend_threshold:
        return "trending_down"

    return "ranging"


def adjust_threshold_for_regime(
    base_threshold: float,
    regime: Regime,
) -> float:
    """Adjust signal threshold based on market regime.

    Strategy:
    - Trending markets: Lower threshold (0.8x) to catch directional moves
    - Ranging markets: Higher threshold (1.2x) to avoid whipsaws
    - High volatility: Higher threshold (1.5x) to avoid false signals

    Args:
        base_threshold: Base signal threshold (e.g., 0.005)
        regime: Current market regime

    Returns:
        Adjusted threshold
    """
    multipliers = {
        "trending_up": 0.8,
        "trending_down": 0.8,
        "ranging": 1.2,
        "high_volatility": 1.5,
    }
    return base_threshold * multipliers.get(regime, 1.0)


def get_regime_description(regime: Regime) -> str:
    """Get human-readable description of regime.

    Args:
        regime: Market regime

    Returns:
        Description string
    """
    descriptions = {
        "trending_up": "Uptrend detected - market showing sustained positive returns",
        "trending_down": "Downtrend detected - market showing sustained negative returns",
        "ranging": "Range-bound market - no clear directional bias",
        "high_volatility": "High volatility regime - elevated risk environment",
    }
    return descriptions.get(regime, "Unknown regime")


def detect_regime_with_details(
    returns: pd.Series | np.ndarray,
    config: RegimeConfig | None = None,
) -> dict[str, float | str]:
    """Detect regime with detailed metrics.

    Args:
        returns: Historical returns
        config: Regime configuration

    Returns:
        Dict with regime and supporting metrics
    """
    if config is None:
        config = RegimeConfig()

    returns = np.asarray(returns)
    regime = detect_regime(returns, config)

    if len(returns) < config.lookback:
        recent_return = float(np.sum(returns)) if len(returns) > 0 else 0.0
        recent_vol = float(np.std(returns)) if len(returns) > 1 else 0.0
    else:
        recent = returns[-config.lookback :]
        recent_return = float(np.sum(recent))
        recent_vol = float(np.std(recent))

    historical_vol = float(np.std(returns)) if len(returns) > 1 else 0.0

    return {
        "regime": regime,
        "description": get_regime_description(regime),
        "cumulative_return": recent_return,
        "recent_volatility": recent_vol,
        "historical_volatility": historical_vol,
        "vol_ratio": recent_vol / historical_vol if historical_vol > 0 else 1.0,
        "lookback": config.lookback,
    }
