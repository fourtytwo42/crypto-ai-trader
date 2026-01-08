"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { rememberMint } from "./RecentMints";

export default function MintSearch() {
  const [mint, setMint] = useState("");
  const router = useRouter();

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const next = mint.trim();
    if (!next) {
      return;
    }
    rememberMint(next);
    router.push(`/mint/${next}`);
  };

  return (
    <form onSubmit={handleSubmit} className="flex w-full max-w-2xl flex-col gap-4 md:flex-row">
      <input
        value={mint}
        onChange={(event) => setMint(event.target.value)}
        placeholder="Paste a pump.fun mint address"
        className="flex-1 rounded-2xl border border-white/60 bg-white/80 px-5 py-3 text-base text-ink shadow-sm focus:border-ink/30 focus:outline-none"
      />
      <button
        type="submit"
        className="rounded-2xl bg-black px-6 py-3 text-sm font-semibold text-white shadow-card transition hover:-translate-y-0.5 hover:bg-black/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-black/40"
      >
        Inspect Mint
      </button>
    </form>
  );
}
