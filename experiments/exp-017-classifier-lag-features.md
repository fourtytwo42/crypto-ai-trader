# Experiment 017 - Classifier with lagged returns

## Goal
Add lagged return features to improve direction accuracy.

## Code changes
- Added `return_lag_1..5` to classifier feature set.
- Computed lagged returns per token in classifier data prep.

## Setup
- Model dir: `models_pumpfun_classifier_exp17`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=15, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp17 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 15 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp17 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.95%`
- Holdout backtest accuracy: `83.85%` (samples=2000)

## Notes
- Lag features did not improve accuracy.
