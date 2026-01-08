# Experiment 011 - Regression higher LR (2e-4)

## Goal
Test if a higher learning rate improves 10-minute price accuracy.

## Setup
- Model dir: `models_pumpfun_sum10_ctx240_lr2e4_e20`
- Horizon: 10 minutes (sum)
- Context length: 240
- Model: NHITS (hidden_size=512, num_layers=3, patch_length=8, stride=4)
- Training: epochs=20, batch_size=16, lr=2e-4
- Holdout count: 12
- Backtest: minutes=10, test_window=120, max_tokens=5, max_samples=500

## Commands
```bash
./venv/bin/python -m src.main pumpfun-train \
  --model-dir models_pumpfun_sum10_ctx240_lr2e4_e20 \
  --horizon-minutes 10 --context-length 240 --model-type nhits --target-mode sum \
  --hidden-size 512 --num-layers 3 --patch-length 8 --stride 4 \
  --epochs 20 --batch-size 16 --learning-rate 2e-4 --holdout-count 12

./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_sum10_ctx240_lr2e4_e20 \
  --minutes 10 --test-window 120 --target-mode sum --max-tokens 5 --max-samples 500
```

## Results
- Train metrics: `mae=0.01929`, `rmse=0.02574`, `mape=2.3619`
- Backtest: `direction_accuracy=6.59%`, `price_accuracy_pct=95.37%`, `mae=0.04663`, `rmse=0.08206`, `samples=440`

## Notes
- No improvement in direction or price accuracy.
