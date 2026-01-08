"""SOL/USD price lookup for pump.fun trades."""

from __future__ import annotations

from datetime import datetime, timezone
import time

import httpx


class SolPriceError(RuntimeError):
    """Raised when SOL price lookup fails."""


def fetch_sol_usd_at(timestamp: int, max_retries: int = 8) -> float:
    """Fetch SOL/USD price using Coingecko for a given unix timestamp."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    start = int(dt.replace(minute=0, second=0, microsecond=0).timestamp())
    end = start + 3600

    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": start,
        "to": end,
    }
    with httpx.Client(timeout=30.0) as client:
        for attempt in range(max_retries):
            response = client.get(url, params=params)
            if response.status_code == 429:
                time.sleep(min(20.0, 0.75 * (2**attempt)))
                continue
            response.raise_for_status()
            data = response.json()
            break
        else:
            response.raise_for_status()

    prices = data.get("prices")
    if not prices:
        raise SolPriceError("No SOL price data returned")

    closest = min(prices, key=lambda item: abs(item[0] / 1000 - timestamp))
    return float(closest[1])
