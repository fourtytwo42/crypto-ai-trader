# Experiment 015 - Classifier without dropout

## Goal
Test if removing dropout improves direction accuracy.

## Setup
- Model dir: `models_pumpfun_classifier_exp15`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.0
- Training: epochs=15, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp15 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.0 \
  --epochs 15 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp15 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.93%`
- Holdout backtest accuracy: `83.25%` (samples=2000)

## Notes
- Dropout removal did not help.
