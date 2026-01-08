"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import useSWR from "swr";

import MintSearch from "../../../components/MintSearch";
import PredictionTable from "../../../components/PredictionTable";
import PriceChart from "../../../components/PriceChart";
import { rememberMint } from "../../../components/RecentMints";
import { fetchCandles, fetchPrediction, fetchToken } from "../../../lib/api";
import { formatPercent, formatTimestamp, formatUsd } from "../../../lib/format";

const REFRESH_MS = 30_000;

export default function MintPage({ params }: { params: { mint: string } }) {
  const mint = useMemo(() => decodeURIComponent(params.mint), [params.mint]);
  const [minutes, setMinutes] = useState(10);
  const [mode, setMode] = useState<"price" | "marketcap">("price");

  useEffect(() => {
    rememberMint(mint);
  }, [mint]);

  const horizon = Math.min(60, Math.max(1, minutes));

  const {
    data: token,
    error: tokenError,
    isLoading: tokenLoading,
  } = useSWR(["token", mint], () => fetchToken(mint), { refreshInterval: REFRESH_MS });

  const {
    data: candles,
    error: candlesError,
    isLoading: candlesLoading,
  } = useSWR(["candles", mint], () => fetchCandles(mint, 240), { refreshInterval: REFRESH_MS });

  const {
    data: prediction,
    error: predictionError,
    isLoading: predictionLoading,
  } = useSWR(["predict", mint, horizon], () => fetchPrediction(mint, horizon), {
    refreshInterval: REFRESH_MS,
  });

  const latestPrediction = prediction?.predictions?.[prediction.predictions.length - 1];
  const currentPrice = token?.current_price ?? prediction?.predictions?.[0]?.current_price ?? null;
  const lastCandle = candles?.[candles.length - 1];
  const projectedValue =
    mode === "marketcap" && latestPrediction
      ? latestPrediction.predicted_price * 1_000_000_000
      : latestPrediction?.predicted_price;

  return (
    <main className="min-h-screen px-6 py-10">
      <section className="mx-auto flex w-full max-w-6xl flex-col gap-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <Link href="/" className="text-sm font-semibold text-ink">
            ← Back to search
          </Link>
          <div className="w-full max-w-xl">
            <MintSearch />
          </div>
        </div>

        <div className="glass-panel relative overflow-hidden rounded-3xl border border-white/60 p-6 shadow-card">
          <div className="hero-sheen" />
          <div className="relative grid gap-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.35em] text-slate">Live Mint</p>
                <h1
                  className="mt-2 text-2xl font-semibold text-ink md:text-3xl"
                  style={{ fontFamily: "var(--font-display)" }}
                >
                  {token?.symbol || mint}
                </h1>
                {token?.name ? <p className="text-sm text-slate">{token.name}</p> : null}
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.3em] text-slate">
                <span className="rounded-full border border-white/60 bg-white/70 px-3 py-1">
                  Refreshes every {REFRESH_MS / 1000}s
                </span>
                <span className="rounded-full border border-white/60 bg-white/70 px-3 py-1">
                  {mint.slice(0, 4)}...{mint.slice(-4)}
                </span>
              </div>
            </div>
            <div className="grid gap-3 md:grid-cols-4">
              <div className="rounded-2xl border border-white/70 bg-white/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">Current price</p>
                <p className="mt-2 text-lg font-semibold text-ink">{formatUsd(currentPrice)}</p>
                <p className="text-xs text-slate">Updated {lastCandle ? lastCandle.timestamp : "--"}</p>
              </div>
              <div className="rounded-2xl border border-white/70 bg-white/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">Market cap</p>
                <p className="mt-2 text-lg font-semibold text-ink">{formatUsd(token?.market_cap_usd)}</p>
                <p className="text-xs text-slate">Last trade {formatTimestamp(token?.last_trade_timestamp)}</p>
              </div>
              <div className="rounded-2xl border border-white/70 bg-white/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">
                  Projected {horizon}m {mode === "marketcap" ? "cap" : "price"}
                </p>
                <p className="mt-2 text-lg font-semibold text-ink">{formatUsd(projectedValue)}</p>
                <p className="text-xs text-slate">Change {formatPercent(latestPrediction?.price_change_pct ?? null)}</p>
              </div>
              <div className="rounded-2xl border border-white/70 bg-white/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-[0.2em] text-slate">Direction</p>
                <p className="mt-2 text-lg font-semibold text-ink">{latestPrediction?.direction || "--"}</p>
                <p className="text-xs text-slate">
                  Conf{" "}
                  {latestPrediction?.direction_confidence !== null &&
                  latestPrediction?.direction_confidence !== undefined
                    ? formatPercent(latestPrediction.direction_confidence * 100)
                    : "--"}
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-8">
          <div className="glass-panel rounded-3xl border border-white/60 p-6 shadow-card">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-slate">Price + Projection</p>
                <h2 className="mt-2 text-2xl font-semibold text-ink">Live candle feed</h2>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex rounded-full border border-white/60 bg-white/70 p-1 text-xs">
                  <button
                    type="button"
                    onClick={() => setMode("price")}
                    className={`rounded-full px-4 py-1 font-semibold ${
                      mode === "price"
                        ? "bg-black text-white shadow-card"
                        : "text-black bg-white/80 hover:bg-white"
                    }`}
                  >
                    Price
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode("marketcap")}
                    className={`rounded-full px-4 py-1 font-semibold ${
                      mode === "marketcap"
                        ? "bg-black text-white shadow-card"
                        : "text-black bg-white/80 hover:bg-white"
                    }`}
                  >
                    Market cap
                  </button>
                </div>
                <label className="text-xs uppercase tracking-[0.3em] text-slate">
                  Minutes
                  <input
                    type="number"
                    min={1}
                    max={60}
                    value={minutes}
                    onChange={(event) => setMinutes(Number(event.target.value))}
                    className="mt-2 w-24 rounded-xl border border-ink/20 bg-white/90 px-3 py-2 text-sm text-ink"
                  />
                </label>
              </div>
            </div>
            <div className="mt-6">
              {candlesLoading ? (
                <div className="h-80 animate-pulse rounded-2xl border border-dashed border-white/70 bg-white/50" />
              ) : candles?.length ? (
                <PriceChart candles={candles} predictions={prediction?.predictions ?? []} mode={mode} />
              ) : (
                <div className="rounded-2xl border border-white/70 bg-white/70 p-6 text-sm text-slate">
                  No candle data yet for this mint.
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-6" />
        </div>
      </section>
    </main>
  );
}
