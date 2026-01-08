# Experiment 006 - Regression: shorter context + higher LR

## Goal
See if a shorter context and larger learning rate improves 10-minute price and direction accuracy.

## Setup
- Model dir: `models_pumpfun_sum10_ctx180_lr5e4_e20`
- Horizon: 10 minutes (sum returns)
- Context length: 180
- Model: NHITS (hidden_size=512, num_layers=3, patch_length=8, stride=4)
- Training: epochs=20, batch_size=16, lr=5e-4
- Holdout count: 12
- Backtest: minutes=10, test_window=120, max_tokens=5, max_samples=500

## Commands
```bash
./venv/bin/python -m src.main pumpfun-train \
  --model-dir models_pumpfun_sum10_ctx180_lr5e4_e20 \
  --horizon-minutes 10 --context-length 180 --model-type nhits --target-mode sum \
  --hidden-size 512 --num-layers 3 --patch-length 8 --stride 4 \
  --epochs 20 --batch-size 16 --learning-rate 5e-4 --holdout-count 12

./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_sum10_ctx180_lr5e4_e20 \
  --minutes 10 --test-window 120 --target-mode sum --max-tokens 5 --max-samples 500
```

## Results
- Train metrics: `mae=0.02033`, `rmse=0.02681`, `mape=1.5222`
- Backtest: `direction_accuracy=7.73%`, `price_accuracy_pct=95.68%`, `mae=0.04285`, `rmse=0.07760`, `samples=440`

## Notes
- Price accuracy similar to baseline; direction accuracy remains poor.
