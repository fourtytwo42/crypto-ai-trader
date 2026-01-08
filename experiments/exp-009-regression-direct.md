# Experiment 009 - Regression direct target (10m)

## Goal
Test direct 10-minute return prediction vs sum-of-returns.

## Setup
- Model dir: `models_pumpfun_direct10_ctx240_lr1e4_e30`
- Horizon: 10 minutes (direct)
- Context length: 240
- Model: NHITS (hidden_size=512, num_layers=3, patch_length=8, stride=4)
- Training: epochs=30, batch_size=16, lr=1e-4
- Holdout count: 12
- Backtest: minutes=10, test_window=120, max_tokens=5, max_samples=500

## Commands
```bash
./venv/bin/python -m src.main pumpfun-train \
  --model-dir models_pumpfun_direct10_ctx240_lr1e4_e30 \
  --horizon-minutes 10 --context-length 240 --model-type nhits --target-mode direct \
  --hidden-size 512 --num-layers 3 --patch-length 8 --stride 4 \
  --epochs 30 --batch-size 16 --learning-rate 1e-4 --holdout-count 12

./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_direct10_ctx240_lr1e4_e30 \
  --minutes 10 --test-window 120 --target-mode direct --max-tokens 5 --max-samples 500
```

## Results
- Train metrics: `mae=3.19`, `rmse=4.23`, `mape=2.96e11`
- Backtest: `direction_accuracy=62.05%`, `price_accuracy_pct=65.74%`, `mae=0.238`, `rmse=1.0897`, `samples=440`

## Notes
- Direct target is substantially worse. Abandon.
