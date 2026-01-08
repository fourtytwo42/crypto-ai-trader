# Experiment 025 - Filtered tokens + threshold

## Goal
Combine min token samples with a small return threshold.

## Setup
- Model dir: `models_pumpfun_classifier_exp25`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=1e-3
- Label threshold: 0.002
- Holdout count: 12
- Min token samples: 300

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp25 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.002 \
  --holdout-count 12 --min-token-samples 300

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp25 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `82.23%`
- Holdout (12 tokens): `84.30%` (samples=5000)

## Notes
- Thresholding hurt accuracy even on filtered tokens.
