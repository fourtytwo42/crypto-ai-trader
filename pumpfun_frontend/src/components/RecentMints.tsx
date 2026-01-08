"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { fetchToken } from "../lib/api";
import { formatPercent, formatUsd } from "../lib/format";
import { TokenSnapshot } from "../lib/types";

const STORAGE_KEY = "pumpfun_recent_mints";

function loadMints(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? (JSON.parse(stored) as string[]) : [];
  } catch {
    return [];
  }
}

function saveMints(mints: string[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(mints));
}

export default function RecentMints() {
  const [mints, setMints] = useState<string[]>([]);
  const [snapshots, setSnapshots] = useState<Record<string, TokenSnapshot>>({});

  useEffect(() => {
    const stored = loadMints();
    setMints(stored);
  }, []);

  useEffect(() => {
    if (!mints.length) return;
    let mounted = true;
    Promise.all(mints.map((mint) => fetchToken(mint).then((data) => [mint, data] as const)))
      .then((entries) => {
        if (!mounted) return;
        const next: Record<string, TokenSnapshot> = {};
        entries.forEach(([mint, data]) => {
          next[mint] = data;
        });
        setSnapshots(next);
      })
      .catch(() => undefined);
    return () => {
      mounted = false;
    };
  }, [mints]);

  const removeMint = (mint: string) => {
    const next = mints.filter((item) => item !== mint);
    setMints(next);
    saveMints(next);
  };

  if (!mints.length) {
    return null;
  }

  return (
    <section className="grid gap-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Watched mints</h2>
        <span className="text-xs uppercase tracking-[0.3em] text-slate">Local</span>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {mints.map((mint) => {
          const data = snapshots[mint];
          return (
            <div key={mint} className="glass-panel rounded-2xl border border-white/70 p-4 shadow-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.3em] text-slate">Mint</p>
                  <p className="mt-1 text-sm font-semibold text-ink">{data?.symbol || mint}</p>
                  <p className="text-xs text-slate line-clamp-1">{mint}</p>
                </div>
                <button
                  onClick={() => removeMint(mint)}
                  className="rounded-full border border-ink/10 px-3 py-1 text-xs text-slate hover:text-ink"
                >
                  Remove
                </button>
              </div>
              <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate">
                <span>Price: {formatUsd(data?.current_price)}</span>
                <span>MC: {formatUsd(data?.market_cap_usd)}</span>
                <span>Completed: {data?.completed ? "Yes" : "No"}</span>
              </div>
              <Link
                href={`/mint/${mint}`}
                className="mt-4 inline-flex items-center text-sm font-semibold text-ink"
              >
                View mint →
              </Link>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function rememberMint(mint: string) {
  const current = loadMints();
  if (current.includes(mint)) return;
  const next = [mint, ...current].slice(0, 12);
  saveMints(next);
}
