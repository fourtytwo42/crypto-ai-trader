# Experiment 013 - Classifier deeper (timeout)

## Goal
Test a deeper classifier (4 layers) for better direction accuracy.

## Setup
- Model dir: `models_pumpfun_classifier_exp13`
- Horizon: 10 minutes
- Architecture: hidden_dim=192, num_layers=4, dropout=0.2
- Training: epochs=12, batch_size=512, lr=8e-4
- Label threshold: 0.0
- Holdout count: 12
- Timeout: 180s

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp13 \
  --horizon-minutes 10 --hidden-dim 192 --num-layers 4 --dropout 0.2 \
  --epochs 12 --batch-size 512 --learning-rate 8e-4 --label-threshold 0.0 --holdout-count 12
```

## Results
- Training timed out at 180s (no metrics captured).

## Notes
- Depth may be too slow; will try smaller variants.
