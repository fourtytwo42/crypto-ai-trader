# Experiment 020 - Classifier batch 256 (timeout)

## Goal
Test smaller batch size for potential generalization improvement.

## Setup
- Model dir: `models_pumpfun_classifier_exp20`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.05
- Training: epochs=15, batch_size=256, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12
- Timeout: 180s

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp20 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.05 \
  --epochs 15 --batch-size 256 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12
```

## Results
- Training timed out at 180s (no metrics captured).

## Notes
- Batch 256 is slower; will retry with fewer epochs.
