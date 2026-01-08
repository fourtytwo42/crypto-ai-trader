# Experiment 044 - Price model (sum target) horizon 1m

## Goal
Evaluate sum-target regression price accuracy at 1 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum01_ctx240_lr1e4_e30`
- Horizon: 1 minute
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
  --model-dir models_pumpfun_sum01_ctx240_lr1e4_e30 \
  --horizon-minutes 1 --context-length 240 --target-mode sum \
  --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 \
  --learning-rate 0.0001 --holdout-count 12

./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_sum01_ctx240_lr1e4_e30 \
  --minutes 1 --test-window 120 --target-mode sum \
  --max-tokens 12 --max-samples 5000
```

## Results
- Holdout (1m):
  - mae=0.0117948
  - rmse=0.0230162
  - smape=194.60
  - direction_accuracy=6.13%
  - price_accuracy_pct=98.67%
  - samples=1190

## Notes
- Price accuracy exceeds 90% at 1m, but direction is very low.
