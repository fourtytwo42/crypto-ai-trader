# Experiment 045 - Price model (sum target) horizon 2m

## Goal
Evaluate sum-target regression price accuracy at 2 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum02_ctx240_lr1e4_e30`
- Horizon: 2 minutes
- Context length: 240
- Model: NHITS
- Target mode: sum
- Hidden size: 256
- Layers: 2
- Epochs: 30
- Batch size: 32
- Learning rate: 1e-4
- Holdout count: 12
- Backtest: test_window=120, max_tokens=12, max_samples=5000

## Commands
```bash
./venv/bin/python -m src.main pumpfun-train \
  --model-dir models_pumpfun_sum02_ctx240_lr1e4_e30 \
  --horizon-minutes 2 --context-length 240 --target-mode sum \
  --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 \
  --learning-rate 0.0001 --holdout-count 12

./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_sum02_ctx240_lr1e4_e30 \
  --minutes 2 --test-window 120 --target-mode sum \
  --max-tokens 12 --max-samples 5000
```

## Results
- Holdout (2m):
  - mae=0.0297966
  - rmse=0.0467390
  - smape=191.89
  - direction_accuracy=9.92%
  - price_accuracy_pct=96.85%
  - samples=1180

## Notes
- Price accuracy remains above 90%.
