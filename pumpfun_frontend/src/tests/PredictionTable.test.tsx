import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PredictionTable from "../components/PredictionTable";
import { Prediction } from "../lib/types";

const predictions: Prediction[] = [
  {
    minutes: 1,
    direction: "UP",
    current_price: 0.00007249,
    predicted_price: 0.0000734,
    price_change: 0.00000091,
    price_change_pct: 1.26,
    direction_confidence: 0.62,
  },
];

describe("PredictionTable", () => {
  it("renders prediction rows", () => {
    render(<PredictionTable predictions={predictions} />);

    expect(screen.getByText("Minute Projections")).toBeInTheDocument();
    expect(screen.getByText("1m")).toBeInTheDocument();
    expect(screen.getByText("UP")).toBeInTheDocument();
    expect(screen.getByText("$0.000073")).toBeInTheDocument();
    expect(screen.getByText("1.26%")).toBeInTheDocument();
    expect(screen.getByText("Conf: 62.00%")).toBeInTheDocument();
  });
});
