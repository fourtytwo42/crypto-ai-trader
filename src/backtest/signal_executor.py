"""Signal execution simulator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Trade:
    """Trade record for backtests."""

    signal: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    return_pct: float


def _apply_costs(raw_return: float, fees: float, slippage: float) -> float:
    return raw_return - 2 * (fees + slippage)


def execute_signals(
    data: pd.DataFrame,
    signals: list[str],
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    fees: float = 0.001,
    slippage: float = 0.0005,
) -> list[Trade]:
    """Execute signals against price data.

    Args:
        data: DataFrame with timestamps and prices.
        signals: List of signals aligned to data rows.
    """
    if len(data) != len(signals):
        raise ValueError("signals length must match data length")
    if price_col not in data.columns or timestamp_col not in data.columns:
        raise ValueError("missing required columns")

    trades: list[Trade] = []
    position: str | None = None
    entry_price = 0.0
    entry_time: datetime | None = None

    for idx, row in data.iterrows():
        signal = signals[idx]
        price = float(row[price_col])
        timestamp = row[timestamp_col]

        if position is None:
            if signal in {"long", "short"}:
                position = signal
                entry_price = price
                entry_time = timestamp
            continue

        if signal != position:
            exit_price = price
            raw_return = (
                (exit_price - entry_price) / entry_price
                if position == "long"
                else (entry_price - exit_price) / entry_price
            )
            net_return = _apply_costs(raw_return, fees, slippage)
            trades.append(
                Trade(
                    signal=position,
                    entry_timestamp=entry_time,
                    exit_timestamp=timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=net_return,
                )
            )
            position = None
            if signal in {"long", "short"}:
                position = signal
                entry_price = price
                entry_time = timestamp

    if position is not None and entry_time is not None:
        last_row = data.iloc[-1]
        exit_price = float(last_row[price_col])
        exit_time = last_row[timestamp_col]
        raw_return = (
            (exit_price - entry_price) / entry_price
            if position == "long"
            else (entry_price - exit_price) / entry_price
        )
        net_return = _apply_costs(raw_return, fees, slippage)
        trades.append(
            Trade(
                signal=position,
                entry_timestamp=entry_time,
                exit_timestamp=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=net_return,
            )
        )

    return trades


def execute_signals_with_holding_period(
    data: pd.DataFrame,
    signals: list[str],
    min_holding_periods: int = 1,
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    fees: float = 0.001,
    slippage: float = 0.0005,
) -> list[Trade]:
    """Execute signals with minimum holding period constraint.

    Prevents exiting a position until it has been held for at least
    `min_holding_periods` periods. This helps avoid overtrading and
    whipsawing in choppy markets.

    Args:
        data: DataFrame with timestamps and prices.
        signals: List of signals aligned to data rows.
        min_holding_periods: Minimum periods to hold before exiting (default 1).
        price_col: Column name for prices.
        timestamp_col: Column name for timestamps.
        fees: Fee rate per trade (default 0.1%).
        slippage: Slippage rate (default 0.05%).

    Returns:
        List of executed trades.
    """
    if len(data) != len(signals):
        raise ValueError("signals length must match data length")
    if price_col not in data.columns or timestamp_col not in data.columns:
        raise ValueError("missing required columns")

    trades: list[Trade] = []
    position: str | None = None
    entry_price = 0.0
    entry_time: datetime | None = None
    periods_held = 0

    for idx, row in data.iterrows():
        signal = signals[idx]
        price = float(row[price_col])
        timestamp = row[timestamp_col]

        if position is None:
            # No position - enter if signal is long or short
            if signal in {"long", "short"}:
                position = signal
                entry_price = price
                entry_time = timestamp
                periods_held = 0
            continue

        # We have a position
        periods_held += 1

        # Only exit if holding period requirement is met and signal changes
        if signal != position and periods_held >= min_holding_periods:
            exit_price = price
            raw_return = (
                (exit_price - entry_price) / entry_price
                if position == "long"
                else (entry_price - exit_price) / entry_price
            )
            net_return = _apply_costs(raw_return, fees, slippage)
            trades.append(
                Trade(
                    signal=position,
                    entry_timestamp=entry_time,
                    exit_timestamp=timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    return_pct=net_return,
                )
            )
            position = None
            # Immediately enter new position if signal is long or short
            if signal in {"long", "short"}:
                position = signal
                entry_price = price
                entry_time = timestamp
                periods_held = 0

    # Close any remaining position at end of data
    if position is not None and entry_time is not None:
        last_row = data.iloc[-1]
        exit_price = float(last_row[price_col])
        exit_time = last_row[timestamp_col]
        raw_return = (
            (exit_price - entry_price) / entry_price
            if position == "long"
            else (entry_price - exit_price) / entry_price
        )
        net_return = _apply_costs(raw_return, fees, slippage)
        trades.append(
            Trade(
                signal=position,
                entry_timestamp=entry_time,
                exit_timestamp=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=net_return,
            )
        )

    return trades
