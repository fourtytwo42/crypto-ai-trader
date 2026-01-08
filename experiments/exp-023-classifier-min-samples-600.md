# Experiment 023 - Classifier filtered to min token samples (600)

## Goal
Push direction accuracy higher by using only higher-activity tokens.

## Setup
- Model dir: `models_pumpfun_classifier_exp23`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 600

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp23 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 600

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp23 \
  --max-tokens 5 --max-samples 2000

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp23 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `84.78%`
- Holdout (5 tokens): `90.75%` (samples=2000)
- Holdout (12 tokens): `88.52%` (samples=5000)

## Notes
- Similar to experiment 022: clears 90% on 5-token sample, below 90% at 12 tokens.
