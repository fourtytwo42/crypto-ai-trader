# Experiment 012 - Baseline regression on 6 holdout tokens

## Goal
Validate baseline price accuracy on a larger sample of holdout tokens.

## Setup
- Model dir: `models_pumpfun_sum10_ctx240_lr1e4_e30`
- Horizon: 10 minutes (sum)
- Backtest: minutes=10, test_window=120, max_tokens=6, max_samples=600

## Command
```bash
./venv/bin/python -m src.main pumpfun-backtest \
  --model-dir models_pumpfun_sum10_ctx240_lr1e4_e30 \
  --minutes 10 --test-window 120 --target-mode sum \
  --max-tokens 6 --max-samples 600
```

## Results
- Backtest: `direction_accuracy=7.82%`, `price_accuracy_pct=95.34%`, `mae=0.04739`, `rmse=0.08364`, `samples=550`

## Notes
- Price accuracy remains strong; direction accuracy remains low when derived from regression.
