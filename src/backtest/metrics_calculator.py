"""Backtest metrics calculator."""

from __future__ import annotations

import math

import numpy as np

from src.backtest.signal_executor import Trade


def calculate_metrics(trades: list[Trade]) -> dict[str, float]:
    """Calculate performance metrics from trades."""
    if not trades:
        return {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_drawdown": 0.0,
            "num_trades": 0,
            "turnover": 0.0,
        }

    returns = np.array([t.return_pct for t in trades], dtype=float)
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / peaks

    total_return = float(equity[-1] - 1)
    sharpe_ratio = float(np.mean(returns) / (np.std(returns) + 1e-12) * math.sqrt(252))
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = float(len(wins) / len(returns))
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    max_drawdown = float(np.min(drawdowns))

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades),
        "turnover": float(len(trades)),
    }


def calculate_prediction_metrics(
    y_true: list[float] | tuple[float, ...],
    y_pred: list[float] | tuple[float, ...],
) -> dict[str, float]:
    """Calculate prediction accuracy and streak metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if not y_true:
        return {
            "directional_accuracy": 0.0,
            "max_correct_streak": 0.0,
            "avg_correct_streak": 0.0,
        }

    def sign(value: float) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    correct = [sign(t) == sign(p) for t, p in zip(y_true, y_pred)]
    directional_accuracy = sum(correct) / len(correct)

    streaks: list[int] = []
    current = 0
    for is_correct in correct:
        if is_correct:
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 0
    if current:
        streaks.append(current)

    max_streak = max(streaks) if streaks else 0
    avg_streak = float(sum(streaks) / len(streaks)) if streaks else 0.0

    return {
        "directional_accuracy": float(directional_accuracy),
        "max_correct_streak": float(max_streak),
        "avg_correct_streak": avg_streak,
    }
