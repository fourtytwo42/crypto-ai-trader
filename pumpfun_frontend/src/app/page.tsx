import MintSearch from "../components/MintSearch";
import RecentMints from "../components/RecentMints";

export default function HomePage() {
  return (
    <main className="min-h-screen px-6 py-12">
      <section className="relative mx-auto flex w-full max-w-5xl flex-col gap-10">
        <div className="glass-panel relative overflow-hidden rounded-3xl border border-white/60 p-10 shadow-card">
          <div className="hero-sheen" />
          <div className="relative space-y-4">
            <p className="text-xs uppercase tracking-[0.4em] text-slate">Pump.fun Signal Lab</p>
            <h1 className="text-4xl font-semibold leading-tight text-ink md:text-5xl" style={{ fontFamily: "var(--font-display)" }}>
              Decode mint momentum in minutes,
              <span className="block text-slate">not hours.</span>
            </h1>
            <p className="max-w-2xl text-base text-slate">
              Paste any mint address to view live candles, market context, and minute-by-minute projections pulled straight
              from your pump.fun models.
            </p>
          </div>
          <div className="mt-8">
            <MintSearch />
          </div>
        </div>
        <RecentMints />
      </section>
    </main>
  );
}
