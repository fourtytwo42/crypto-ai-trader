import { describe, expect, it, vi, beforeEach } from "vitest";

import { apiUrl, checkHealth, fetchForecast, listModels } from "@/lib/api";

describe("api client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("builds api url", () => {
    expect(apiUrl("/health")).toContain("/health");
  });

  it("fetches models list", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => [{ id: 1, name: "nhits-demo", model_type: "nhits", version: "1", created_at: "2024-01-01" }]
    } as Response);

    const models = await listModels();
    expect(models).toHaveLength(1);
    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("fetches forecast with payload", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        symbol: "BTC-USDT",
        hours: 24,
        model_id: 1,
        model_name: "nhits-demo",
        data_timestamp: "2024-01-01",
        target_timestamp: "2024-01-02",
        predicted_at: "2024-01-01",
        current_price: 40000,
        predicted_price: 41000,
        price_change: 1000,
        price_change_pct: 2.5,
        direction: "UP",
        cache_hit: false
      })
    } as Response);

    const forecast = await fetchForecast("BTC-USDT", 24, 1, false);
    expect(forecast.symbol).toBe("BTC-USDT");
    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("checks api health", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok" })
    } as Response);

    const health = await checkHealth();
    expect(health.status).toBe("ok");
    expect(fetchSpy).toHaveBeenCalledOnce();
  });
});
