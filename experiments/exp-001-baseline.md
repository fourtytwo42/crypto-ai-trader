# Experiment 001 - Baseline pump.fun models (sampled holdout)

## Goal
Establish current baseline for 10-minute price accuracy and direction accuracy on holdout tokens.

## Setup
- Model (price): `models_pumpfun_sum10_ctx240_lr1e4_e30`
- Model (direction): `models_pumpfun_classifier_v9`
- Target: 10-minute horizon
- Holdout scope: sampled for speed (`max_tokens=5`, `max_samples=500` for price; `max_tokens=5`, `max_samples=2000` for direction)

## Commands
```bash
./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_sum10_ctx240_lr1e4_e30 \
  --minutes 10 --test-window 120 --target-mode sum \
  --max-tokens 5 --max-samples 500

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_v9 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Price backtest metrics: `mae=0.04553`, `rmse=0.07909`, `smape=194.43`, `direction_accuracy=5.68%`, `price_accuracy_pct=95.50%`, `samples=440`
- Direction classifier backtest: `directional_accuracy=84.6%`, `samples=2000`

## Notes
- Price accuracy remains strong; direction accuracy from regression is poor.
- Classifier is closer but still below 90% on this sampled holdout.
- Full holdout backtest needs longer runtime; will run after promising configs surface.
