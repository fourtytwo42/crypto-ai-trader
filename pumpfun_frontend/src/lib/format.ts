export function formatUsd(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  if (value >= 1_000_000_000) {
    return `$${(value / 1_000_000_000).toFixed(2)}B`;
  }
  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(2)}M`;
  }
  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(2)}K`;
  }
  return `$${value.toFixed(6)}`;
}

export function formatPercent(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(2)}%`;
}

export function formatTimestamp(ms?: number | null): string {
  if (!ms) {
    return "--";
  }
  const date = new Date(ms > 1_000_000_000_000 ? ms : ms * 1000);
  return date.toLocaleString();
}
