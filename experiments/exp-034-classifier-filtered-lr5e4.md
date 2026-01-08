# Experiment 034 - Filtered tokens, lower LR

## Goal
Test if a lower learning rate helps on filtered tokens.

## Setup
- Model dir: `models_pumpfun_classifier_exp34`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=5e-4
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 300

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp34 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 5e-4 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp34 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `84.88%`
- Holdout (12 tokens): `88.52%` (samples=5000)

## Notes
- No improvement.
