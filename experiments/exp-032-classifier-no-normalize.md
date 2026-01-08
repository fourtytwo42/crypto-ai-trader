# Experiment 032 - Classifier without normalization

## Goal
Test if raw features improve direction accuracy.

## Code changes
- Added `--no-normalize` to disable feature normalization for classifier.

## Setup
- Model dir: `models_pumpfun_classifier_exp32`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 300
- Normalization: disabled

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp32 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300 --no-normalize

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp32 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `81.38%`
- Holdout (12 tokens): `88.80%` (samples=5000)

## Notes
- No improvement.
