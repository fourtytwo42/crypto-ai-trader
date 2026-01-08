import { test, expect } from "@playwright/test";

const models = [
  {
    id: 1,
    name: "nhits-core-ctx336",
    model_type: "nhits",
    version: "1.0",
    created_at: "2024-01-01T00:00:00Z",
    metrics: { mae: 0.6, rmse: 0.7 }
  }
];

const history = [
  {
    symbol: "BTC-USDT",
    hours: 24,
    model_id: 1,
    model_name: "nhits-core-ctx336",
    data_timestamp: "2024-01-01T00:00:00Z",
    target_timestamp: "2024-01-02T00:00:00Z",
    predicted_at: "2024-01-01T00:00:00Z",
    predicted_price: 41000,
    actual_price: 40500,
    accuracy_pct: 0.5
  }
];

test.beforeEach(async ({ page }) => {
  await page.route("**/health", async (route) => {
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.route("**/models", async (route) => {
    await route.fulfill({ json: models });
  });
  await page.route("**/trainings", async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({
        json: {
          id: 11,
          status: "queued",
          model_name: "nhits-core-ctx336",
          created_at: "2024-01-01T00:00:00Z"
        }
      });
      return;
    }
    await route.fulfill({ json: [] });
  });
  await page.route("**/forecast/history/**", async (route) => {
    await route.fulfill({ json: history });
  });
  await page.route("**/forecast/predict", async (route) => {
    const body = route.request().postDataJSON();
    await route.fulfill({
      json: {
        symbol: body.symbol,
        hours: 24,
        model_id: 1,
        model_name: "nhits-core-ctx336",
        data_timestamp: "2024-01-01T00:00:00Z",
        target_timestamp: "2024-01-02T00:00:00Z",
        predicted_at: "2024-01-01T00:00:00Z",
        current_price: 40000,
        predicted_price: 41000,
        price_change: 1000,
        price_change_pct: 2.5,
        direction: "UP",
        cache_hit: body.use_cache
      }
    });
  });
});

test("renders forecast cards and refresh button", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("Market intelligence")).toBeVisible();
  await expect(page.getByText("BTC-USDT forecast")).toBeVisible();

  const refreshButton = page.getByRole("button", { name: "Refresh all forecasts" });
  await expect(refreshButton).toBeEnabled();
  await refreshButton.click();

  await expect(page.getByText("Prediction history")).toBeVisible();
});
