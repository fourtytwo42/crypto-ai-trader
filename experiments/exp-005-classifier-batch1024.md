# Experiment 005 - Wider batch, deeper model

## Goal
Stabilize training with a larger batch and slightly deeper network.

## Setup
- Model dir: `models_pumpfun_classifier_exp05`
- Horizon: 10 minutes
- Architecture: hidden_dim=96, num_layers=3, dropout=0.05
- Training: epochs=12, batch_size=1024, lr=8e-4
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp05 \
  --horizon-minutes 10 --hidden-dim 96 --num-layers 3 --dropout 0.05 \
  --epochs 12 --batch-size 1024 --learning-rate 8e-4 \
  --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp05 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.99%`
- Holdout backtest accuracy: `84.25%` (samples=2000)

## Notes
- No improvement over baseline.
