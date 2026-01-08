# Experiment 008 - Classifier with higher threshold (0.01)

## Goal
Increase directional accuracy by filtering small moves.

## Setup
- Model dir: `models_pumpfun_classifier_exp08`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=15, batch_size=512, lr=1e-3
- Label threshold: 0.01
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp08 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 15 --batch-size 512 --learning-rate 1e-3 \
  --label-threshold 0.01 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp08 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `82.28%`
- Holdout backtest accuracy: `82.20%` (samples=2000)

## Notes
- Larger notice threshold worsened accuracy.
