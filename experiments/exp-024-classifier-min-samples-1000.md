# Experiment 024 - Classifier filtered to min token samples (1000)

## Goal
Push direction accuracy by using only the most active tokens.

## Setup
- Model dir: `models_pumpfun_classifier_exp24`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 1000

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp24 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 1000

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp24 \
  --max-tokens 5 --max-samples 2000

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp24 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `84.65%`
- Holdout (5 tokens): `90.60%` (samples=2000)
- Holdout (12 tokens): `88.52%` (samples=5000)

## Notes
- Still below 90% once expanded to 12 tokens.
