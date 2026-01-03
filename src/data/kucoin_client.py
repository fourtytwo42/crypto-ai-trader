"""KuCoin REST client for candle backfill."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import httpx


BASE_URL = "https://api.kucoin.com/api/v1/market/candles"

TF_TO_SECONDS = {
    "1min": 60,
    "3min": 3 * 60,
    "5min": 5 * 60,
    "15min": 15 * 60,
    "30min": 30 * 60,
    "1hour": 3600,
    "2hour": 2 * 3600,
    "4hour": 4 * 3600,
    "6hour": 6 * 3600,
    "8hour": 8 * 3600,
    "12hour": 12 * 3600,
    "1day": 86400,
    "1week": 7 * 86400,
}


class KuCoinError(RuntimeError):
    """Error raised for KuCoin API issues."""


@dataclass(frozen=True)
class KuCoinCandle:
    """Normalized candle data from KuCoin."""

    timestamp: int
    open: float
    close: float
    high: float
    low: float
    volume: float
    turnover: float


def _fetch_chunk(
    client: httpx.Client,
    symbol: str,
    timeframe: str,
    start_at: int,
    end_at: int,
) -> list[KuCoinCandle]:
    params = {
        "symbol": symbol,
        "type": timeframe,
        "startAt": start_at,
        "endAt": end_at,
    }
    resp = client.get(BASE_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise KuCoinError("Unexpected KuCoin response format")

    candles: list[KuCoinCandle] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 7:
            raise KuCoinError("Invalid candle row from KuCoin")
        ts, op, cl, hi, lo, vol, turnover = row[:7]
        candles.append(
            KuCoinCandle(
                timestamp=int(ts),
                open=float(op),
                close=float(cl),
                high=float(hi),
                low=float(lo),
                volume=float(vol),
                turnover=float(turnover),
            )
        )
    return candles


def backfill_kucoin_candles(
    symbol: str,
    timeframe: str,
    start_at: int,
    end_at: int | None = None,
    sleep_seconds: float = 0.2,
) -> Iterable[KuCoinCandle]:
    """Backfill KuCoin candles over a time range.

    Args:
        symbol: KuCoin symbol (e.g., BTC-USDT).
        timeframe: KuCoin timeframe string (e.g., 1hour).
        start_at: Start timestamp (inclusive, unix seconds).
        end_at: End timestamp (inclusive, unix seconds). Defaults to now.
        sleep_seconds: Delay between requests to avoid rate limits.

    Yields:
        KuCoinCandle objects in chronological order.
    """
    if timeframe not in TF_TO_SECONDS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    now = int(time.time())
    current_end = end_at or now
    start_at = int(start_at)

    window = 1500 * TF_TO_SECONDS[timeframe]

    with httpx.Client() as client:
        while current_end > start_at:
            current_start = max(start_at, current_end - window)
            candles = _fetch_chunk(client, symbol, timeframe, current_start, current_end)

            if not candles:
                current_end = current_start - 1
                time.sleep(sleep_seconds)
                continue

            # KuCoin returns newest->oldest; yield oldest->newest.
            candles.reverse()
            for candle in candles:
                yield candle

            oldest_ts = candles[0].timestamp
            current_end = oldest_ts - 1
            time.sleep(sleep_seconds)
