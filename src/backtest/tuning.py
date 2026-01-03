"""Backtest tuning harness with best-practice metric scoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from src.config import get_settings
from src.data.pipeline import prepare_training_data
from src.database.connection import get_db_manager
from src.training.config import TrainingConfig


@dataclass
class TuningResult:
    """Result for a backtest configuration."""

    score: float
    metrics: dict[str, float]
    config: dict[str, float | bool]


def _score_metrics(metrics: dict[str, float]) -> float:
    """Score metrics with Sharpe and drawdown penalty."""
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    max_drawdown = float(metrics.get("max_drawdown", 0.0))
    num_trades = float(metrics.get("num_trades", 0.0))

    if num_trades < 5:
        return -math.inf

    return sharpe - 2.0 * abs(max_drawdown)


def run_grid(
    model_dir: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    epochs: int,
    force_simple: bool,
) -> list[TuningResult]:
    """Run a small grid search over signal settings."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    config = TrainingConfig(**metadata["config"])
    config.epochs = epochs

    settings = get_settings()
    db = get_db_manager()
    with db.session() as session:
        data, _ = prepare_training_data(session, normalize=False)

    data = data.dropna().reset_index(drop=True)

    thresholds = [0.0025, 0.0035]
    max_spreads = [0.02]
    volatility_adaptive_values = [True]
    uncertainty_enabled_values = [True]

    results: list[TuningResult] = []
    for threshold in thresholds:
        for max_spread in max_spreads:
            for volatility_adaptive in volatility_adaptive_values:
                for uncertainty_enabled in uncertainty_enabled_values:
                    backtest_config = BacktestConfig(
                        threshold=threshold,
                        volatility_adaptive=volatility_adaptive,
                        volatility_window=settings.signal_volatility_window,
                        max_uncertainty_spread=max_spread,
                        uncertainty_enabled=uncertainty_enabled,
                        min_holding_periods=1,
                    )
                    result = walk_forward_backtest(
                        data,
                        config,
                        train_size=train_size,
                        val_size=val_size,
                        test_size=test_size,
                        step_size=step_size,
                        model_dir=str(model_dir),
                        force_simple=force_simple,
                        backtest_config=backtest_config,
                    )
                    metrics = result.aggregated
                    score = _score_metrics(metrics)
                    results.append(
                        TuningResult(
                            score=score,
                            metrics=metrics,
                            config={
                                "threshold": threshold,
                                "max_uncertainty_spread": max_spread,
                                "volatility_adaptive": volatility_adaptive,
                                "uncertainty_enabled": uncertainty_enabled,
                            },
                        )
                    )

    results.sort(key=lambda item: item.score, reverse=True)
    return results


if __name__ == "__main__":
    top = run_grid(
        model_dir="models",
        train_size=800,
        val_size=100,
        test_size=100,
        step_size=400,
        epochs=3,
        force_simple=True,
    )
    for item in top[:5]:
        print(
            {
                "score": item.score,
                "config": item.config,
                "metrics": item.metrics,
            }
        )
