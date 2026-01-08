# Experiment 010 - Classifier threshold 0.002

## Goal
Try a mild return threshold.

## Setup
- Model dir: `models_pumpfun_classifier_exp10`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=1e-3
- Label threshold: 0.002
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp10 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 1e-3 \
  --label-threshold 0.002 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp10 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `82.16%`
- Holdout backtest accuracy: `82.20%` (samples=2000)

## Notes
- No improvement.
