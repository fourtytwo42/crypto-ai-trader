import { Candle, PredictResponse, TokenSnapshot } from "./types";

const baseUrl = process.env.NEXT_PUBLIC_PUMPFUN_API_URL || "http://127.0.0.1:8081";

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`);
  if (!res.ok) {
    throw new Error(`API ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchToken(mint: string): Promise<TokenSnapshot> {
  return request(`/token?mint=${encodeURIComponent(mint)}`);
}

export async function fetchCandles(mint: string, limit = 240): Promise<Candle[]> {
  const data = await request<{ candles: Candle[] }>(
    `/candles?mint=${encodeURIComponent(mint)}&limit=${limit}`
  );
  return data.candles;
}

export function fetchPrediction(mint: string, minutes: number): Promise<PredictResponse> {
  return request(`/predict?mint=${encodeURIComponent(mint)}&minutes=${minutes}`);
}
