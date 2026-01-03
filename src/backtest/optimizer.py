"""Backtest-driven optimization loop for signal settings."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import structlog

from src.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from src.config import get_settings
from src.data.pipeline import prepare_training_data
from src.database.connection import get_db_manager
from src.training.config import TrainingConfig

logger = structlog.get_logger(__name__)

@dataclass
class OptimizationResult:
    """Optimization result for a single configuration."""

    score: float
    metrics: dict[str, float]
    config: dict[str, float | bool]


def score_metrics(metrics: dict[str, float]) -> float:
    """Score based on forecast accuracy (lower is better)."""
    mae = float(metrics.get("mae", 0.0))
    rmse = float(metrics.get("rmse", 0.0))
    mape = float(metrics.get("mape", 0.0))
    return -(mae + rmse + mape)


def run_optimization(
    model_dir: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    epochs: int,
    horizon: int = 12,
    max_rounds: int = 6,
    patience: int = 2,
) -> list[OptimizationResult]:
    """Run repeated grid searches until no improvement."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    config = TrainingConfig(**metadata["config"])
    config.epochs = epochs
    config.horizon = horizon

    settings = get_settings()
    db = get_db_manager()
    with db.session() as session:
        data, _ = prepare_training_data(session, normalize=False)

    if "return" in data.columns:
        data = data.copy()
        data["target_return"] = data["return"].shift(-horizon)
    data = data.dropna().reset_index(drop=True)

    grids = [
        {
            "context_length": [128, 168],
            "hidden_size": [256, 512],
            "num_layers": [4, 6],
            "patch_length": [16],
            "stride": [8],
            "learning_rate": [0.0001, 0.0005],
        }
    ]

    all_results: list[OptimizationResult] = []
    best_score = -math.inf
    stale_rounds = 0

    for round_idx in range(max_rounds):
        grid = grids[min(round_idx, len(grids) - 1)]
        round_best = best_score
        total_configs = (
            len(grid["context_length"])
            * len(grid["hidden_size"])
            * len(grid["num_layers"])
            * len(grid["patch_length"])
            * len(grid["stride"])
            * len(grid["learning_rate"])
        )
        config_idx = 0
        logger.info(
            "Optimization round start",
            round=round_idx + 1,
            total_rounds=max_rounds,
            total_configs=total_configs,
            best_score=best_score,
        )

        for context_length in grid["context_length"]:
            for hidden_size in grid["hidden_size"]:
                for num_layers in grid["num_layers"]:
                    for patch_length in grid["patch_length"]:
                        for stride in grid["stride"]:
                            for learning_rate in grid["learning_rate"]:
                                config_idx += 1
                                logger.info(
                                    "Running config",
                                    round=round_idx + 1,
                                    config_index=config_idx,
                                    total_configs=total_configs,
                                    context_length=context_length,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    patch_length=patch_length,
                                    stride=stride,
                                    learning_rate=learning_rate,
                                )
                                config.context_length = context_length
                                config.hidden_size = hidden_size
                                config.num_layers = num_layers
                                config.patch_length = patch_length
                                config.stride = stride
                                config.learning_rate = learning_rate

                                backtest_config = BacktestConfig(
                                    threshold=settings.signal_threshold,
                                    volatility_adaptive=True,
                                    volatility_window=settings.signal_volatility_window,
                                    max_uncertainty_spread=settings.signal_max_uncertainty_spread,
                                    uncertainty_enabled=True,
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
                                    backtest_config=backtest_config,
                                )
                                metrics = result.aggregated
                                score = score_metrics(metrics)
                                all_results.append(
                                    OptimizationResult(
                                        score=score,
                                        metrics=metrics,
                                        config={
                                            "context_length": context_length,
                                            "hidden_size": hidden_size,
                                            "num_layers": num_layers,
                                            "patch_length": patch_length,
                                            "stride": stride,
                                            "learning_rate": learning_rate,
                                        },
                                    )
                                )
                                logger.info(
                                    "Config result",
                                    round=round_idx + 1,
                                    config_index=config_idx,
                                    score=score,
                                    metrics=metrics,
                                    best_score=best_score,
                                )
                                if score > round_best:
                                    round_best = score
                                if score > best_score:
                                    best_score = score
                                    logger.info(
                                        "New best score",
                                        round=round_idx + 1,
                                        config_index=config_idx,
                                        best_score=best_score,
                                    )

        if round_best <= best_score:
            stale_rounds += 1
        else:
            stale_rounds = 0

        logger.info(
            "Optimization round end",
            round=round_idx + 1,
            round_best=round_best,
            best_score=best_score,
            stale_rounds=stale_rounds,
        )
        if stale_rounds >= patience:
            logger.info("Stopping optimization due to no improvement", patience=patience)
            break

    all_results.sort(key=lambda item: item.score, reverse=True)
    return all_results


if __name__ == "__main__":
    results = run_optimization(
        model_dir="models",
        train_size=1200,
        val_size=200,
        test_size=200,
        step_size=200,
        epochs=50,
    )
    for item in results[:5]:
        print(
            {
                "score": item.score,
                "config": item.config,
                "metrics": item.metrics,
            }
        )
