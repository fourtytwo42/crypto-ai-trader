import { formatDateTime } from "@/lib/format";
import type { ModelInfo } from "@/lib/api";

const metricLabelMap: Record<string, string> = {
  mae: "MAE",
  rmse: "RMSE",
  mape: "MAPE"
};

export default function ModelSelector({
  models,
  selectedModelId,
  onSelect,
  isLoading,
  error
}: {
  models: ModelInfo[];
  selectedModelId: number | null;
  onSelect: (id: number) => void;
  isLoading?: boolean;
  error?: string | null;
}) {
  const activeModel = models.find((model) => model.id === selectedModelId) ?? models[0];

  return (
    <div className="glass-card rounded-3xl px-6 py-8 shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-slate">Model control</p>
          <h2 className="mt-2 text-2xl font-semibold text-ink">Choose the active model</h2>
          <p className="mt-2 text-sm text-slate">
            Compare metrics, swap architectures, and see how each model shapes the forecast.
          </p>
        </div>
        <div className="min-w-[220px]">
          <select
            className="w-full rounded-2xl border border-white/70 bg-white/80 px-4 py-3 text-sm font-semibold text-ink shadow-halo focus:border-ink/30 focus:outline-none"
            value={activeModel?.id ?? ""}
            onChange={(event) => onSelect(Number(event.target.value))}
            disabled={isLoading || models.length === 0}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error ? (
        <p className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </p>
      ) : null}

      {activeModel ? (
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-4">
            <p className="text-xs uppercase tracking-[0.2em] text-slate">Type</p>
            <p className="mt-2 text-lg font-semibold text-ink">{activeModel.model_type}</p>
            <p className="text-xs text-slate">Version {activeModel.version}</p>
          </div>
          <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-4">
            <p className="text-xs uppercase tracking-[0.2em] text-slate">Created</p>
            <p className="mt-2 text-lg font-semibold text-ink">
              {formatDateTime(activeModel.created_at)}
            </p>
          </div>
          <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-4">
            <p className="text-xs uppercase tracking-[0.2em] text-slate">Metrics</p>
            <div className="mt-2 space-y-1 text-sm text-slate">
              {activeModel.metrics ? (
                Object.entries(activeModel.metrics).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <span>{metricLabelMap[key] ?? key}</span>
                    <span className="font-semibold text-ink">
                      {typeof value === "number" ? value.toFixed(3) : String(value)}
                    </span>
                  </div>
                ))
              ) : (
                <span>No metrics recorded</span>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6 rounded-2xl border border-white/70 bg-white/60 px-4 py-4 text-sm text-slate">
          {isLoading ? "Loading models..." : "No models available yet."}
        </div>
      )}
    </div>
  );
}
