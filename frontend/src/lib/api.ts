export type ModelInfo = {
  id: number;
  name: string;
  model_type: string;
  version: string;
  created_at: string;
  metrics?: Record<string, unknown> | null;
};

export type ForecastPredictResponse = {
  symbol: string;
  hours: number;
  model_id: number;
  model_name: string;
  data_timestamp: string;
  target_timestamp: string;
  predicted_at: string;
  current_price: number;
  predicted_price: number;
  price_change: number;
  price_change_pct: number;
  direction: string;
  cache_hit: boolean;
  actual_price?: number | null;
  accuracy_pct?: number | null;
};

export type ForecastHistoryItem = {
  symbol: string;
  hours: number;
  model_id: number;
  model_name: string;
  data_timestamp: string;
  target_timestamp: string;
  predicted_at: string;
  predicted_price: number;
  actual_price?: number | null;
  accuracy_pct?: number | null;
};

export type TrainingJobResponse = {
  id: number;
  status: string;
  model_name: string;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  metrics?: Record<string, unknown> | null;
  error?: string | null;
  model_id?: number | null;
};

export type TrainingRequest = {
  model_type: string;
  context_length: number;
  horizon_hours: number;
  hidden_size: number;
  num_layers: number;
  patch_length: number;
  stride: number;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  loss_type: string;
  device?: string | null;
  symbols?: string[] | null;
  multi_asset: boolean;
  name_prefix?: string | null;
  max_vram_gb?: number | null;
};

export type HealthResponse = {
  status: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    headers: {
      "Content-Type": "application/json"
    },
    ...init
  });

  if (!response.ok) {
    let details = "";
    try {
      const body = await response.json();
      details = body?.detail?.message || response.statusText;
    } catch (error) {
      details = response.statusText;
    }
    throw new Error(`API ${response.status}: ${details}`);
  }

  return response.json() as Promise<T>;
}

export function listModels() {
  return apiFetch<ModelInfo[]>("/models");
}

export function fetchForecast(
  symbol: string,
  hours: number,
  modelId?: number | null,
  useCache = true
) {
  return apiFetch<ForecastPredictResponse>("/forecast/predict", {
    method: "POST",
    body: JSON.stringify({
      symbol,
      hours,
      model_id: modelId ?? undefined,
      use_cache: useCache
    })
  });
}

export function fetchForecastHistory(
  symbol: string,
  modelId?: number | null,
  limit = 80
) {
  const query = new URLSearchParams();
  if (modelId) query.set("model_id", String(modelId));
  query.set("limit", String(limit));
  return apiFetch<ForecastHistoryItem[]>(`/forecast/history/${symbol}?${query.toString()}`);
}

export function listTrainings() {
  return apiFetch<TrainingJobResponse[]>("/trainings");
}

export function startTraining(request: TrainingRequest) {
  return apiFetch<TrainingJobResponse>("/trainings", {
    method: "POST",
    body: JSON.stringify(request)
  });
}

export function checkHealth() {
  return apiFetch<HealthResponse>("/health");
}
