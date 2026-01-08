export type TokenSnapshot = {
  mint: string;
  token_id: string;
  symbol?: string | null;
  name?: string | null;
  created_timestamp: number;
  king_of_the_hill_timestamp?: number | null;
  completed: boolean;
  current_price?: number | null;
  market_cap_usd?: number | null;
  last_trade_timestamp?: number | null;
};

export type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume_usd: number;
  trades: number;
};

export type Prediction = {
  minutes: number;
  direction: "UP" | "DOWN" | "FLAT";
  current_price: number;
  predicted_price: number;
  price_change: number;
  price_change_pct: number;
  direction_confidence?: number | null;
};

export type PredictResponse = {
  mint: string;
  token_id: string;
  model_dir: string;
  horizon: number;
  predictions: Prediction[];
};
