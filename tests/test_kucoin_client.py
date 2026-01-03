"""Tests for KuCoin candle backfill client."""

from __future__ import annotations

from typing import Any

import pytest

from src.data import kucoin_client


class DummyResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_fetch_chunk_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse KuCoin response into normalized candles."""

    def fake_get(*_args, **_kwargs):
        return DummyResponse(
            {
                "code": "200000",
                "data": [
                    ["2", "10", "11", "12", "9", "100", "200"],
                    ["1", "9", "10", "11", "8", "90", "180"],
                ],
            }
        )

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def get(self, *args, **kwargs):  # noqa: ANN001
            return fake_get(*args, **kwargs)

    monkeypatch.setattr(kucoin_client.httpx, "Client", lambda: DummyClient())

    with kucoin_client.httpx.Client() as client:
        candles = kucoin_client._fetch_chunk(
            client, "BTC-USDT", "1hour", start_at=1, end_at=2
        )

    assert len(candles) == 2
    assert candles[0].timestamp == 2
    assert candles[0].open == 10.0
    assert candles[1].timestamp == 1
    assert candles[1].close == 10.0


def test_backfill_pages_and_reverses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backfill respects paging and reverses newest->oldest."""
    calls: list[tuple[int, int]] = []

    def fake_fetch_chunk(_client, _symbol, _tf, start_at, end_at):
        calls.append((start_at, end_at))
        if len(calls) == 1:
            return [
                kucoin_client.KuCoinCandle(300, 1, 1, 1, 1, 1, 1),
                kucoin_client.KuCoinCandle(200, 1, 1, 1, 1, 1, 1),
                kucoin_client.KuCoinCandle(100, 1, 1, 1, 1, 1, 1),
            ]
        if len(calls) == 2:
            return [
                kucoin_client.KuCoinCandle(50, 1, 1, 1, 1, 1, 1),
                kucoin_client.KuCoinCandle(40, 1, 1, 1, 1, 1, 1),
            ]
        return []

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

    monkeypatch.setattr(kucoin_client, "_fetch_chunk", fake_fetch_chunk)
    monkeypatch.setattr(kucoin_client.httpx, "Client", lambda: DummyClient())
    monkeypatch.setattr(kucoin_client.time, "sleep", lambda *_: None)

    results = list(
        kucoin_client.backfill_kucoin_candles(
            symbol="BTC-USDT",
            timeframe="1hour",
            start_at=0,
            end_at=400,
            sleep_seconds=0,
        )
    )

    assert calls[0][1] == 400
    assert calls[1][1] == 99
    assert [c.timestamp for c in results[:3]] == [100, 200, 300]
    assert [c.timestamp for c in results[3:]] == [40, 50]
