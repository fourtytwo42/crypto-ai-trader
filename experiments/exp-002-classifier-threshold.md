# Experiment 002 - Direction classifier with return threshold

## Goal
Improve 10-minute direction accuracy by removing near-zero returns.

## Setup
- Model dir: `models_pumpfun_classifier_exp02`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=15, batch_size=512, lr=1e-3
- Label threshold: 0.001 (drop near-zero returns)
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp02 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 15 --batch-size 512 --learning-rate 1e-3 \
  --label-threshold 0.001 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp02 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `82.12%`
- Holdout backtest accuracy: `83.0%` (samples=2000)

## Notes
- Thresholding did not improve direction accuracy on sampled holdout.
- Next idea: increase context features / add regime features or adjust normalization window.
