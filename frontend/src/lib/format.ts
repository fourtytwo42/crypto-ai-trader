export function formatCurrency(value: number, currency = "USD") {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2
  }).format(value);
}

export function formatNumber(value: number, maximumFractionDigits = 2) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits
  }).format(value);
}

export function formatPercent(value: number, maximumFractionDigits = 2) {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    maximumFractionDigits
  }).format(value / 100);
}

export function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

export function formatShortDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit"
  }).format(date);
}

export function directionBadge(direction: string) {
  if (direction === "UP") return "badge-up";
  if (direction === "DOWN") return "badge-down";
  return "badge-flat";
}
