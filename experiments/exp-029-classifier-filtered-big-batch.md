# Experiment 029 - Filtered tokens + larger batch

## Goal
Try larger batch size on filtered tokens.

## Setup
- Model dir: `models_pumpfun_classifier_exp29`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=1024, lr=8e-4
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 300

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp29 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 8 --batch-size 1024 --learning-rate 8e-4 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp29 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `84.77%`
- Holdout (12 tokens): `88.16%` (samples=5000)

## Notes
- Larger batch did not help.
