# Experiment 021 - Classifier batch 256, fewer epochs

## Goal
Retry batch size 256 with fewer epochs.

## Setup
- Model dir: `models_pumpfun_classifier_exp21`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.05
- Training: epochs=8, batch_size=256, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp21 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.05 \
  --epochs 8 --batch-size 256 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp21 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.94%`
- Holdout backtest accuracy: `83.85%` (samples=2000)

## Notes
- No improvement.
