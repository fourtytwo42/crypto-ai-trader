# Experiment 019 - Shallow classifier

## Goal
Try a single-layer classifier (logistic-style).

## Setup
- Model dir: `models_pumpfun_classifier_exp19`
- Horizon: 10 minutes
- Architecture: hidden_dim=64, num_layers=1, dropout=0.0
- Training: epochs=15, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp19 \
  --horizon-minutes 10 --hidden-dim 64 --num-layers 1 --dropout 0.0 \
  --epochs 15 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp19 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.51%`
- Holdout backtest accuracy: `83.75%` (samples=2000)

## Notes
- Shallow model still below baseline.
