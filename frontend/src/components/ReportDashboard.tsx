"use client";

import { useMemo, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { RefreshCcw, Activity, Sparkles } from "lucide-react";

import { DEFAULT_HOURS, DEFAULT_SYMBOLS } from "@/lib/constants";
import {
  checkHealth,
  fetchForecast,
  fetchForecastHistory,
  listModels,
  listTrainings,
  startTraining,
  type ForecastHistoryItem,
  type ForecastPredictResponse,
  type ModelInfo,
  type TrainingJobResponse,
  type TrainingRequest
} from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import ModelSelector from "@/components/ModelSelector";
import PredictionCard from "@/components/PredictionCard";
import TrainingPanel from "@/components/TrainingPanel";

export default function ReportDashboard() {
  const { mutate } = useSWRConfig();
  const [selectedModelId, setSelectedModelId] = useState<number | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [trainingMessage, setTrainingMessage] = useState<string | null>(null);

  const { data: health } = useSWR("health", checkHealth, {
    refreshInterval: 60_000
  });
  const {
    data: models,
    isLoading: modelsLoading,
    error: modelsError
  } = useSWR<ModelInfo[]>("models", listModels);
  const { data: trainings } = useSWR<TrainingJobResponse[]>("trainings", listTrainings, {
    refreshInterval: 30_000
  });

  const activeModel = useMemo(() => {
    if (!models || models.length === 0) return null;
    return models.find((model) => model.id === selectedModelId) ?? models[0];
  }, [models, selectedModelId]);

  const forecastData = DEFAULT_SYMBOLS.map((symbol) => {
    const forecastKey = activeModel ? ["forecast", symbol, activeModel.id] : null;
    const historyKey = activeModel ? ["history", symbol, activeModel.id] : null;

    const forecast = useSWR<ForecastPredictResponse>(
      forecastKey,
      () => fetchForecast(symbol, DEFAULT_HOURS, activeModel?.id ?? null, true),
      {
        revalidateOnFocus: false
      }
    );

    const history = useSWR<ForecastHistoryItem[]>(
      historyKey,
      () => fetchForecastHistory(symbol, activeModel?.id ?? null, 80),
      {
        revalidateOnFocus: false
      }
    );

    return {
      symbol,
      forecast,
      history
    };
  });

  const refreshAll = async () => {
    if (!activeModel) return;
    setIsRefreshing(true);
    setTrainingMessage(null);
    await Promise.all(
      DEFAULT_SYMBOLS.map(async (symbol) => {
        await mutate(
          ["forecast", symbol, activeModel.id],
          () => fetchForecast(symbol, DEFAULT_HOURS, activeModel.id, false),
          { revalidate: false }
        );
        await mutate(["history", symbol, activeModel.id]);
      })
    );
    setLastRefresh(new Date());
    setIsRefreshing(false);
  };

  const handleTraining = async (request: TrainingRequest) => {
    setTrainingMessage(null);
    try {
      const response = await startTraining(request);
      setTrainingMessage(`Training job queued: ${response.model_name}`);
      await mutate("trainings");
      await mutate("models");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Training failed";
      setTrainingMessage(message);
    }
  };

  return (
    <section className="relative mx-auto flex max-w-7xl flex-col gap-10 px-6 pb-16 pt-12">
      <header className="glass-card relative overflow-hidden rounded-3xl border border-white/50 px-8 py-10 shadow-card">
        <div className="absolute right-6 top-6 hidden h-24 w-24 rounded-full bg-mint/20 blur-3xl lg:block" />
        <div className="absolute bottom-6 left-6 hidden h-24 w-24 rounded-full bg-sun/30 blur-3xl lg:block" />
        <div className="relative flex flex-col gap-6">
          <div className="flex items-center gap-3 text-sm text-slate">
            <span className="flex items-center gap-2 rounded-full border border-white/60 bg-white/60 px-3 py-1 shadow-halo">
              <Activity className="h-4 w-4" />
              {health?.status === "ok" ? "API online" : "API offline"}
            </span>
            <span className="rounded-full border border-white/60 bg-white/50 px-3 py-1 shadow-halo">
              {activeModel ? `Model: ${activeModel.name}` : "No model selected"}
            </span>
            {lastRefresh ? (
              <span className="rounded-full border border-white/60 bg-white/50 px-3 py-1 shadow-halo">
                Refreshed {formatDateTime(lastRefresh.toISOString())}
              </span>
            ) : null}
          </div>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="font-[var(--font-display)] text-4xl leading-tight text-ink md:text-5xl">
                Market intelligence,
                <span className="block text-slate">streamed straight from your models.</span>
              </p>
              <p className="mt-4 max-w-2xl text-lg text-slate">
                Track live 24-hour forecasts, price deltas, and accuracy signals across the top
                crypto pairs. Trigger retraining with a single click and compare model performance
                as the market moves.
              </p>
            </div>
            <button
              type="button"
              onClick={refreshAll}
              className="inline-flex items-center gap-2 rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white shadow-glow transition hover:-translate-y-0.5 hover:shadow-card"
              disabled={isRefreshing || !activeModel}
            >
              <RefreshCcw className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
              {isRefreshing ? "Refreshing" : "Refresh all forecasts"}
            </button>
          </div>
        </div>
      </header>

      <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="flex flex-col gap-8">
          <ModelSelector
            models={models ?? []}
            selectedModelId={activeModel?.id ?? null}
            onSelect={setSelectedModelId}
            isLoading={modelsLoading}
            error={modelsError instanceof Error ? modelsError.message : null}
          />

          <div className="grid gap-6">
            {forecastData.map(({ symbol, forecast, history }) => (
              <PredictionCard
                key={symbol}
                symbol={symbol}
                forecast={forecast.data}
                history={history.data}
                isLoading={forecast.isLoading || history.isLoading}
                error={(forecast.error || history.error) as Error | undefined}
              />
            ))}
          </div>
        </div>

        <aside className="flex flex-col gap-8">
          <TrainingPanel onSubmit={handleTraining} trainings={trainings ?? []} />
          <div className="glass-card rounded-3xl px-6 py-8 shadow-card">
            <div className="flex items-center gap-3">
              <Sparkles className="h-5 w-5 text-sun" />
              <h3 className="text-lg font-semibold">System Notes</h3>
            </div>
            <div className="mt-4 space-y-3 text-sm text-slate">
              <p>
                Forecasts are aligned with the quick-predict pathway and the model loader now
                falls back to CPU when a GPU backend isnt available.
              </p>
              <p>
                Use refresh when you want fresh KuCoin data and new cached forecasts. Training
                jobs run asynchronously, so check the status panel as they complete.
              </p>
              {trainingMessage ? (
                <p className="rounded-2xl border border-white/70 bg-white/70 px-4 py-3 text-ink">
                  {trainingMessage}
                </p>
              ) : null}
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}
