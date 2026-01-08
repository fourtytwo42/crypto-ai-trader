import { TrendingDown, TrendingUp, Minus } from "lucide-react";

import type { ForecastHistoryItem, ForecastPredictResponse } from "@/lib/api";
import {
  directionBadge,
  formatCurrency,
  formatDateTime,
  formatNumber,
  formatPercent
} from "@/lib/format";
import HistoryChart from "@/components/HistoryChart";

const directionIconMap = {
  UP: TrendingUp,
  DOWN: TrendingDown,
  FLAT: Minus
};

export default function PredictionCard({
  symbol,
  forecast,
  history,
  isLoading,
  error
}: {
  symbol: string;
  forecast?: ForecastPredictResponse;
  history?: ForecastHistoryItem[];
  isLoading?: boolean;
  error?: Error;
}) {
  const DirectionIcon = directionIconMap[forecast?.direction as keyof typeof directionIconMap] ?? Minus;

  return (
    <article className="glass-card rounded-3xl px-6 py-8 shadow-card">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate">{symbol}</p>
          <h3 className="mt-2 text-2xl font-semibold text-ink">{symbol} forecast</h3>
        </div>
        <div
          className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold uppercase ${
            forecast ? directionBadge(forecast.direction) : "badge-flat"
          }`}
        >
          <DirectionIcon className="h-4 w-4" />
          {forecast?.direction ?? "PENDING"}
        </div>
      </div>

      {error ? (
        <p className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error.message}
        </p>
      ) : null}

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-white/70 bg-white/70 px-4 py-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate">Current price</p>
          <p className="mt-2 text-2xl font-semibold text-ink">
            {forecast ? formatCurrency(forecast.current_price) : "--"}
          </p>
          <p className="text-xs text-slate">
            Updated {forecast ? formatDateTime(forecast.data_timestamp) : "--"}
          </p>
        </div>
        <div className="rounded-2xl border border-white/70 bg-white/70 px-4 py-4">
          <p className="text-xs uppercase tracking-[0.2em] text-slate">Predicted price</p>
          <p className="mt-2 text-2xl font-semibold text-ink">
            {forecast ? formatCurrency(forecast.predicted_price) : "--"}
          </p>
          <p className="text-xs text-slate">
            Target {forecast ? formatDateTime(forecast.target_timestamp) : "--"}
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-3">
          <p className="text-xs uppercase tracking-[0.2em] text-slate">Change</p>
          <p className="mt-1 text-lg font-semibold text-ink">
            {forecast ? formatCurrency(forecast.price_change) : "--"}
          </p>
          <p className="text-xs text-slate">
            {forecast ? formatPercent(forecast.price_change_pct) : "--"}
          </p>
        </div>
        <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-3">
          <p className="text-xs uppercase tracking-[0.2em] text-slate">Accuracy</p>
          <p className="mt-1 text-lg font-semibold text-ink">
            {forecast?.accuracy_pct != null ? `${formatNumber(forecast.accuracy_pct, 2)}%` : "--"}
          </p>
          <p className="text-xs text-slate">Based on latest actual close</p>
        </div>
        <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-3">
          <p className="text-xs uppercase tracking-[0.2em] text-slate">Cache</p>
          <p className="mt-1 text-lg font-semibold text-ink">
            {forecast ? (forecast.cache_hit ? "Cache hit" : "Live run") : "--"}
          </p>
          <p className="text-xs text-slate">Predicted {forecast ? formatDateTime(forecast.predicted_at) : "--"}</p>
        </div>
      </div>

      <div className="chart-surface mt-6 rounded-2xl px-4 py-4">
        <div className="flex items-center justify-between text-xs uppercase tracking-[0.2em] text-slate">
          <span>Prediction history</span>
          <span>{history?.length ?? 0} runs</span>
        </div>
        <div className="mt-4">
          {isLoading ? (
            <div className="flex h-40 items-center justify-center text-sm text-slate">
              Loading forecast history...
            </div>
          ) : (
            <HistoryChart history={history ?? []} />
          )}
        </div>
      </div>
    </article>
  );
}
