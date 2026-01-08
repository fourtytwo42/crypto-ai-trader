# Experiment 014 - Narrow classifier

## Goal
See if a smaller classifier generalizes better.

## Setup
- Model dir: `models_pumpfun_classifier_exp14`
- Horizon: 10 minutes
- Architecture: hidden_dim=64, num_layers=2, dropout=0.1
- Training: epochs=20, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp14 \
  --horizon-minutes 10 --hidden-dim 64 --num-layers 2 --dropout 0.1 \
  --epochs 20 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp14 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.96%`
- Holdout backtest accuracy: `83.90%` (samples=2000)

## Notes
- Slightly below baseline.
