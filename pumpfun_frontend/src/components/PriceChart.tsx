"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

import { Candle, Prediction } from "../lib/types";

interface PriceChartProps {
  candles: Candle[];
  predictions: Prediction[];
  mode: "price" | "marketcap";
}

const SUPPLY = 1_000_000_000;

export default function PriceChart({ candles, predictions, mode }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      layout: {
        background: { type: ColorType.Solid, color: "#f4f3ee" },
        textColor: "#0c0f14",
      },
      grid: {
        vertLines: { color: "rgba(12, 15, 20, 0.08)" },
        horzLines: { color: "rgba(12, 15, 20, 0.08)" },
      },
      height: 320,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const priceFormat =
      mode === "marketcap"
        ? { type: "price", precision: 2, minMove: 0.01 }
        : { type: "price", precision: 8, minMove: 0.00000001 };

    const priceSeries = chart.addLineSeries({
      color: "#0c0f14",
      lineWidth: 2,
      priceFormat,
    });
    priceSeries.setData(
      candles.map((candle) => ({
        time: Math.floor(new Date(candle.timestamp).getTime() / 1000),
        value: mode === "marketcap" ? candle.close * SUPPLY : candle.close,
      }))
    );

    if (predictions.length) {
      const lastTime = new Date(candles[candles.length - 1].timestamp).getTime();
      const projection = predictions.map((item) => ({
        time: Math.floor((lastTime + item.minutes * 60_000) / 1000),
        value: mode === "marketcap" ? item.predicted_price * SUPPLY : item.predicted_price,
      }));
      const projectionSeries = chart.addLineSeries({
        color: "#3ad5ff",
        lineWidth: 2,
        lineStyle: 2,
        priceFormat,
      });
      projectionSeries.setData(projection);
    }

    const resize = () => {
      chart.applyOptions({ width: containerRef.current?.clientWidth || 600 });
    };
    resize();
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      chart.remove();
    };
  }, [candles, predictions, mode]);

  return <div ref={containerRef} className="h-80 w-full" />;
}
