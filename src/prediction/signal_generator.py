"""Signal generation logic with volatility-adaptive thresholds and uncertainty bands.

Provides enhanced signal generation that:
1. Scales threshold based on recent volatility (volatility-adaptive)
2. Filters trades based on prediction uncertainty (q90-q10 spread)
3. Optionally calculates position size based on conviction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Signal = Literal["long", "short", "flat"]


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    base_threshold: float = 0.005
    volatility_adaptive: bool = True
    baseline_volatility: float = 0.02
    volatility_multiplier_min: float = 0.5
    volatility_multiplier_max: float = 2.0
    uncertainty_enabled: bool = True
    max_uncertainty_spread: float = 0.03
    asymmetric_long_multiplier: float = 1.0
    asymmetric_short_multiplier: float = 1.0


def calculate_effective_threshold(
    base_threshold: float,
    current_volatility: float,
    baseline_volatility: float,
    multiplier_min: float = 0.5,
    multiplier_max: float = 2.0,
) -> float:
    """Calculate volatility-scaled threshold.

    Formula: effective_threshold = base_threshold * clamp(current_vol / baseline_vol, min, max)

    When volatility is higher than baseline, threshold increases (more selective).
    When volatility is lower than baseline, threshold decreases (less selective).

    Args:
        base_threshold: Base threshold (e.g., 0.005 = 0.5%)
        current_volatility: Current volatility measure (e.g., ret_std_7)
        baseline_volatility: Baseline volatility for scaling (~0.02 for daily BTC)
        multiplier_min: Minimum scaling factor (floor)
        multiplier_max: Maximum scaling factor (ceiling)

    Returns:
        Scaled effective threshold
    """
    if baseline_volatility <= 0:
        return base_threshold

    raw_multiplier = current_volatility / baseline_volatility
    clamped_multiplier = max(multiplier_min, min(multiplier_max, raw_multiplier))
    return base_threshold * clamped_multiplier


def calculate_uncertainty_spread(prediction: dict[str, float]) -> float:
    """Calculate prediction uncertainty as q90 - q10 spread.

    A wider spread indicates more uncertainty in the prediction.
    This represents the 80% confidence interval width.

    Args:
        prediction: Dict with q10, q50, q90 quantile predictions

    Returns:
        The spread between q90 and q10
    """
    return float(prediction["q90"]) - float(prediction["q10"])


def calculate_conviction(prediction: dict[str, float]) -> float:
    """Calculate conviction score as |q50| / spread.

    Higher conviction means the median prediction is large relative to uncertainty.
    Used for dynamic position sizing.

    Args:
        prediction: Dict with q10, q50, q90 quantile predictions

    Returns:
        Conviction score between 0.0 and 1.0
    """
    q50 = abs(float(prediction["q50"]))
    spread = calculate_uncertainty_spread(prediction)
    if spread <= 0:
        return 1.0  # Perfect certainty (degenerate case)
    return min(1.0, q50 / spread)


def generate_signal(
    prediction: dict[str, float],
    threshold: float = 0.005,
    current_volatility: float | None = None,
    config: SignalConfig | None = None,
) -> Signal:
    """Generate a trading signal from quantile prediction.

    Enhanced signal generation with:
    1. Volatility-adaptive threshold scaling
    2. Uncertainty-based no-trade filtering
    3. Asymmetric long/short thresholds

    Args:
        prediction: Dict with q10, q50, q90 quantile predictions
        threshold: Base threshold (used if config not provided)
        current_volatility: Current volatility for adaptive threshold
        config: Full signal configuration (overrides threshold if provided)

    Returns:
        Trading signal: "long", "short", or "flat"
    """
    if config is None:
        config = SignalConfig(base_threshold=threshold)

    q10 = float(prediction["q10"])
    q50 = float(prediction["q50"])
    q90 = float(prediction["q90"])

    # Step 1: Calculate effective threshold (volatility-adaptive)
    if config.volatility_adaptive and current_volatility is not None:
        effective_threshold = calculate_effective_threshold(
            config.base_threshold,
            current_volatility,
            config.baseline_volatility,
            config.volatility_multiplier_min,
            config.volatility_multiplier_max,
        )
    else:
        effective_threshold = config.base_threshold

    # Step 2: Check uncertainty spread (no-trade band)
    if config.uncertainty_enabled:
        spread = calculate_uncertainty_spread(prediction)
        if spread > config.max_uncertainty_spread:
            return "flat"  # Too uncertain, don't trade

    # Step 3: Apply asymmetric thresholds
    long_threshold = effective_threshold * config.asymmetric_long_multiplier
    short_threshold = effective_threshold * config.asymmetric_short_multiplier

    # Step 4: Generate signal with enhanced logic
    # Long: median above threshold AND 10th percentile not too negative
    if q50 > long_threshold and q10 > -long_threshold:
        return "long"
    # Short: median below negative threshold AND 90th percentile not too positive
    if q50 < -short_threshold and q90 < short_threshold:
        return "short"
    return "flat"


def generate_signal_with_position_size(
    prediction: dict[str, float],
    threshold: float = 0.005,
    current_volatility: float | None = None,
    config: SignalConfig | None = None,
    min_size: float = 0.1,
    max_size: float = 1.0,
) -> tuple[Signal, float]:
    """Generate signal with dynamic position size based on conviction.

    Position size scales with conviction: higher |q50| relative to spread
    results in larger position size.

    Args:
        prediction: Dict with q10, q50, q90 quantile predictions
        threshold: Base threshold
        current_volatility: Current volatility for adaptive threshold
        config: Signal configuration
        min_size: Minimum position size as fraction (default 0.1 = 10%)
        max_size: Maximum position size as fraction (default 1.0 = 100%)

    Returns:
        Tuple of (signal, position_size)
    """
    signal = generate_signal(prediction, threshold, current_volatility, config)

    if signal == "flat":
        return signal, 0.0

    conviction = calculate_conviction(prediction)
    position_size = min_size + conviction * (max_size - min_size)
    return signal, position_size


def apply_signals(
    predictions: list[dict[str, float]],
    threshold: float = 0.005,
    volatility_series: list[float] | None = None,
    config: SignalConfig | None = None,
) -> list[Signal]:
    """Apply signal generation to a list of predictions.

    Args:
        predictions: List of quantile predictions (each with q10, q50, q90)
        threshold: Base threshold
        volatility_series: Volatility values aligned with predictions (optional)
        config: Signal configuration

    Returns:
        List of signals aligned with input predictions
    """
    signals: list[Signal] = []
    for i, pred in enumerate(predictions):
        current_vol = volatility_series[i] if volatility_series else None
        signals.append(generate_signal(pred, threshold, current_vol, config))
    return signals


def apply_signals_with_sizing(
    predictions: list[dict[str, float]],
    threshold: float = 0.005,
    volatility_series: list[float] | None = None,
    config: SignalConfig | None = None,
    min_size: float = 0.1,
    max_size: float = 1.0,
) -> list[tuple[Signal, float]]:
    """Apply signal generation with position sizing to a list of predictions.

    Args:
        predictions: List of quantile predictions
        threshold: Base threshold
        volatility_series: Volatility values aligned with predictions
        config: Signal configuration
        min_size: Minimum position size
        max_size: Maximum position size

    Returns:
        List of (signal, position_size) tuples
    """
    results: list[tuple[Signal, float]] = []
    for i, pred in enumerate(predictions):
        current_vol = volatility_series[i] if volatility_series else None
        results.append(
            generate_signal_with_position_size(
                pred, threshold, current_vol, config, min_size, max_size
            )
        )
    return results
