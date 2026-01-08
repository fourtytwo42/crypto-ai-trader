import { expect, test } from "@playwright/test";

test("search navigates to mint view and renders projections", async ({ page }) => {
  const mint = "BB76w3GdSXobKJEzf2GZssWeHu1DwiQmrAJrsXBspump";

  await page.route("**/token?*", async (route) => {
    await route.fulfill({
      json: {
        mint,
        token_id: "token_123",
        symbol: "BUMP",
        name: "Bump Token",
        created_timestamp: 1_720_000_000,
        completed: false,
        current_price: 0.00007249,
        market_cap_usd: 120_000,
        last_trade_timestamp: 1_720_000_500,
      },
    });
  });

  await page.route("**/candles?*", async (route) => {
    await route.fulfill({
      json: {
        candles: [
          {
            timestamp: "2026-01-08T05:00:00Z",
            open: 0.00007,
            high: 0.000074,
            low: 0.000069,
            close: 0.000072,
            volume_usd: 1400,
            trades: 22,
          },
          {
            timestamp: "2026-01-08T05:01:00Z",
            open: 0.000072,
            high: 0.000075,
            low: 0.000071,
            close: 0.0000724,
            volume_usd: 1900,
            trades: 31,
          },
        ],
      },
    });
  });

  await page.route("**/predict?*", async (route) => {
    await route.fulfill({
      json: {
        mint,
        token_id: "token_123",
        model_dir: "/models/regression",
        horizon: 10,
        predictions: [
          {
            minutes: 1,
            direction: "UP",
            current_price: 0.00007249,
            predicted_price: 0.0000734,
            price_change: 0.00000091,
            price_change_pct: 1.26,
            direction_confidence: 0.62,
          },
        ],
      },
    });
  });

  await page.goto("/");
  await page.getByPlaceholder("Paste a pump.fun mint address").fill(mint);
  await page.getByRole("button", { name: "Inspect Mint" }).click();

  await expect(page).toHaveURL(`/mint/${mint}`);
  await expect(page.getByText("Live candle feed")).toBeVisible();
  await expect(page.getByText("Minute Projections")).toBeVisible();
  await expect(page.getByText("Projected 10m")).toBeVisible();
});
