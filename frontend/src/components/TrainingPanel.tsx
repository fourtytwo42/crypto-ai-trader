"use client";

import { useState } from "react";
import { Rocket } from "lucide-react";

import type { TrainingJobResponse, TrainingRequest } from "@/lib/api";
import { DEFAULT_SYMBOLS } from "@/lib/constants";
import { formatDateTime } from "@/lib/format";

const defaultRequest: TrainingRequest = {
  model_type: "nhits",
  context_length: 336,
  horizon_hours: 24,
  hidden_size: 512,
  num_layers: 3,
  patch_length: 8,
  stride: 4,
  learning_rate: 0.00005,
  batch_size: 16,
  epochs: 50,
  loss_type: "huber",
  multi_asset: true,
  symbols: DEFAULT_SYMBOLS
};

export default function TrainingPanel({
  onSubmit,
  trainings
}: {
  onSubmit: (request: TrainingRequest) => Promise<void> | void;
  trainings: TrainingJobResponse[];
}) {
  const [form, setForm] = useState(defaultRequest);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const updateForm = (updates: Partial<TrainingRequest>) => {
    setForm((prev) => ({ ...prev, ...updates }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSubmitting(true);
    await onSubmit(form);
    setIsSubmitting(false);
  };

  return (
    <div className="glass-card rounded-3xl px-6 py-8 shadow-card">
      <div className="flex items-center gap-3">
        <Rocket className="h-5 w-5 text-mint" />
        <h3 className="text-lg font-semibold">Train a fresh model</h3>
      </div>
      <p className="mt-2 text-sm text-slate">
        Kick off a new training job and monitor its status. Defaults mirror the production
        quick-predict settings.
      </p>

      <form className="mt-6 grid gap-4" onSubmit={handleSubmit}>
        <div className="grid gap-3 md:grid-cols-2">
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Model type
            <select
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
              value={form.model_type}
              onChange={(event) => updateForm({ model_type: event.target.value })}
            >
              <option value="nhits">NHITS</option>
              <option value="patchtst">PatchTST</option>
            </select>
          </label>
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Context length
            <input
              type="number"
              min={24}
              max={720}
              value={form.context_length}
              onChange={(event) => updateForm({ context_length: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Hidden size
            <input
              type="number"
              min={64}
              max={2048}
              value={form.hidden_size}
              onChange={(event) => updateForm({ hidden_size: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Layers
            <input
              type="number"
              min={1}
              max={24}
              value={form.num_layers}
              onChange={(event) => updateForm({ num_layers: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Patch length
            <input
              type="number"
              min={4}
              max={64}
              value={form.patch_length}
              onChange={(event) => updateForm({ patch_length: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Stride
            <input
              type="number"
              min={1}
              max={32}
              value={form.stride}
              onChange={(event) => updateForm({ stride: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Epochs
            <input
              type="number"
              min={1}
              max={500}
              value={form.epochs}
              onChange={(event) => updateForm({ epochs: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Learning rate
            <input
              type="number"
              step={0.00001}
              min={0.00001}
              max={0.01}
              value={form.learning_rate}
              onChange={(event) => updateForm({ learning_rate: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs uppercase tracking-[0.2em] text-slate">
            Batch size
            <input
              type="number"
              min={8}
              max={256}
              value={form.batch_size}
              onChange={(event) => updateForm({ batch_size: Number(event.target.value) })}
              className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
            />
          </label>
        </div>

        <div>
          <label className="text-xs uppercase tracking-[0.2em] text-slate">Symbols</label>
          <input
            type="text"
            value={(form.symbols ?? []).join(",")}
            onChange={(event) =>
              updateForm({
                symbols: event.target.value
                  .split(",")
                  .map((symbol) => symbol.trim().toUpperCase())
                  .filter(Boolean)
              })
            }
            className="mt-2 w-full rounded-2xl border border-white/60 bg-white/80 px-3 py-2 text-sm"
          />
        </div>

        <label className="flex items-center gap-3 text-sm text-slate">
          <input
            type="checkbox"
            checked={form.multi_asset}
            onChange={(event) => updateForm({ multi_asset: event.target.checked })}
            className="h-4 w-4"
          />
          Train as multi-asset model
        </label>

        <button
          type="submit"
          disabled={isSubmitting}
          className="mt-2 inline-flex items-center justify-center rounded-2xl bg-ink px-4 py-3 text-sm font-semibold text-white shadow-glow transition hover:-translate-y-0.5"
        >
          {isSubmitting ? "Starting training..." : "Start training"}
        </button>
      </form>

      <div className="mt-8">
        <p className="text-xs uppercase tracking-[0.3em] text-slate">Recent training jobs</p>
        <div className="mt-3 space-y-3 text-sm">
          {trainings.length === 0 ? (
            <div className="rounded-2xl border border-white/60 bg-white/60 px-4 py-3 text-slate">
              No training jobs yet.
            </div>
          ) : (
            trainings.slice(0, 4).map((job) => (
              <div
                key={job.id}
                className="rounded-2xl border border-white/60 bg-white/70 px-4 py-3"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink">{job.model_name}</span>
                  <span className="text-xs uppercase tracking-[0.2em] text-slate">
                    {job.status}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate">
                  Created {formatDateTime(job.created_at)}
                </p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
