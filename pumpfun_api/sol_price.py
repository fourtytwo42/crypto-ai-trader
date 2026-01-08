from __future__ import annotations

import httpx


def fetch_sol_usd_at(timestamp: int) -> float:
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart/range"
    params = {
        "vs_currency": "usd",
        "from": timestamp - 600,
        "to": timestamp + 600,
    }
    with httpx.Client(timeout=10.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError("No SOL price data returned")
    closest = min(prices, key=lambda item: abs(item[0] / 1000 - timestamp))
    return float(closest[1])
