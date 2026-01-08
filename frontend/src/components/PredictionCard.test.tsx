import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PredictionCard from "@/components/PredictionCard";

const forecast = {
  symbol: "BTC-USDT",
  hours: 24,
  model_id: 1,
  model_name: "nhits-demo",
  data_timestamp: "2024-01-01T00:00:00Z",
  target_timestamp: "2024-01-02T00:00:00Z",
  predicted_at: "2024-01-01T00:00:00Z",
  current_price: 40000,
  predicted_price: 41000,
  price_change: 1000,
  price_change_pct: 2.5,
  direction: "UP",
  cache_hit: false,
  actual_price: null,
  accuracy_pct: null
};

const history = [
  {
    symbol: "BTC-USDT",
    hours: 24,
    model_id: 1,
    model_name: "nhits-demo",
    data_timestamp: "2024-01-01T00:00:00Z",
    target_timestamp: "2024-01-02T00:00:00Z",
    predicted_at: "2024-01-01T00:00:00Z",
    predicted_price: 41000,
    actual_price: 40500,
    accuracy_pct: 0.5
  }
];

describe("PredictionCard", () => {
  it("renders forecast details", () => {
    render(
      <PredictionCard
        symbol="BTC-USDT"
        forecast={forecast}
        history={history}
        isLoading={false}
      />
    );

    expect(screen.getByText("BTC-USDT forecast")).toBeInTheDocument();
    expect(screen.getByText("Current price")).toBeInTheDocument();
    expect(screen.getByText("Predicted price")).toBeInTheDocument();
  });
});
