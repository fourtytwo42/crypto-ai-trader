import { describe, expect, it } from "vitest";

import { directionBadge, formatCurrency, formatPercent } from "@/lib/format";

describe("format helpers", () => {
  it("formats currency", () => {
    expect(formatCurrency(1200)).toBe("$1,200.00");
  });

  it("formats percent from delta", () => {
    expect(formatPercent(12.5)).toBe("12.5%");
  });

  it("maps direction to badge class", () => {
    expect(directionBadge("UP")).toBe("badge-up");
    expect(directionBadge("DOWN")).toBe("badge-down");
    expect(directionBadge("FLAT")).toBe("badge-flat");
  });
});
