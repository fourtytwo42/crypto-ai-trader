import { formatPercent, formatUsd } from "../lib/format";
import { Prediction } from "../lib/types";

const SUPPLY = 1_000_000_000;

type DisplayMode = "price" | "marketcap";

function DirectionIcon({ direction }: { direction: Prediction["direction"] }) {
  if (direction === "UP") {
    return (
      <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-neon/20 text-neon">
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 19V5" />
          <path d="m5 12 7-7 7 7" />
        </svg>
      </span>
    );
  }
  if (direction === "DOWN") {
    return (
      <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-lava/20 text-lava">
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 5v14" />
          <path d="m19 12-7 7-7-7" />
        </svg>
      </span>
    );
  }
  return (
    <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-slate/20 text-slate">
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M5 12h14" />
      </svg>
    </span>
  );
}

export default function PredictionTable({
  predictions,
  mode,
}: {
  predictions: Prediction[];
  mode: DisplayMode;
}) {
  return (
    <div className="rounded-2xl border border-white/60 bg-white/70 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Minute Projections</h3>
        <span className="text-xs uppercase tracking-[0.3em] text-slate">Rolling</span>
      </div>
      <div className="mt-4 grid gap-2 text-sm">
        {predictions.map((item) => (
          <div
            key={item.minutes}
            className="grid grid-cols-[64px_auto] gap-3 rounded-xl border border-white/60 bg-white/80 px-3 py-3"
          >
            <div className="flex flex-col items-start justify-center gap-2">
              <span className="text-xs uppercase tracking-[0.2em] text-slate">{item.minutes}m</span>
              <DirectionIcon direction={item.direction} />
            </div>
            <div className="grid gap-2 md:grid-cols-3">
              <div>
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">
                  {mode === "price" ? "Projected price" : "Projected cap"}
                </p>
                <p className="text-sm font-semibold text-ink">
                  {formatUsd(mode === "price" ? item.predicted_price : item.predicted_price * SUPPLY)}
                </p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">Change</p>
                <p className="text-sm font-semibold text-ink">{formatPercent(item.price_change_pct)}</p>
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">Direction conf</p>
                <p className="text-sm font-semibold text-ink">
                  {item.direction_confidence !== null && item.direction_confidence !== undefined
                    ? formatPercent(item.direction_confidence * 100)
                    : "--"}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
