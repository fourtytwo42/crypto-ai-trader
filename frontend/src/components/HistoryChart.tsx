"use client";

import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { formatCurrency, formatShortDate } from "@/lib/format";
import type { ForecastHistoryItem } from "@/lib/api";

const tooltipStyles = {
  borderRadius: "12px",
  border: "1px solid rgba(12, 18, 22, 0.1)",
  backgroundColor: "rgba(255, 255, 255, 0.9)",
  boxShadow: "0 12px 30px -24px rgba(12, 18, 22, 0.5)",
  fontSize: "12px"
} as const;

export default function HistoryChart({ history }: { history: ForecastHistoryItem[] }) {
  const data = history
    .slice()
    .reverse()
    .map((item) => ({
      date: formatShortDate(item.predicted_at),
      predicted: item.predicted_price,
      actual: item.actual_price ?? null
    }));

  if (data.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-sm text-slate">
        Prediction history will appear after the first forecast run.
      </div>
    );
  }

  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
          <XAxis dataKey="date" tickLine={false} axisLine={false} />
          <YAxis
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => formatCurrency(Number(value))}
            width={70}
          />
          <Tooltip
            formatter={(value: number) => formatCurrency(Number(value))}
            contentStyle={tooltipStyles}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#0b1114"
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#3dd5b0"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
