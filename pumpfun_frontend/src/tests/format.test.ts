import { describe, expect, it } from "vitest";

import { formatPercent, formatUsd } from "../lib/format";

describe("formatUsd", () => {
  it("formats null values", () => {
    expect(formatUsd(null)).toBe("--");
  });

  it("formats large numbers", () => {
    expect(formatUsd(2_400_000)).toBe("$2.40M");
    expect(formatUsd(1_250_000_000)).toBe("$1.25B");
  });

  it("formats small numbers", () => {
    expect(formatUsd(0.00007249)).toBe("$0.000072");
  });
});

describe("formatPercent", () => {
  it("formats null values", () => {
    expect(formatPercent(undefined)).toBe("--");
  });

  it("formats percentages", () => {
    expect(formatPercent(3.456)).toBe("3.46%");
  });
});
