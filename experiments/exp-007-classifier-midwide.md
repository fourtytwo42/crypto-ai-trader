# Experiment 007 - Classifier mid-width 160

## Goal
Try a mid-width, mid-depth classifier for 10-minute direction.

## Setup
- Model dir: `models_pumpfun_classifier_exp07`
- Horizon: 10 minutes
- Architecture: hidden_dim=160, num_layers=3, dropout=0.15
- Training: epochs=12, batch_size=512, lr=7e-4
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp07 \
  --horizon-minutes 10 --hidden-dim 160 --num-layers 3 --dropout 0.15 \
  --epochs 12 --batch-size 512 --learning-rate 7e-4 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp07 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.94%`
- Holdout backtest accuracy: `83.65%` (samples=2000)

## Notes
- Still below baseline.
