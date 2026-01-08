# Experiment 028 - Filtered tokens + deeper model (8 epochs)

## Goal
Try a deeper classifier on filtered tokens with fewer epochs.

## Setup
- Model dir: `models_pumpfun_classifier_exp28`
- Horizon: 10 minutes
- Architecture: hidden_dim=256, num_layers=3, dropout=0.2
- Training: epochs=8, batch_size=512, lr=8e-4
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 300

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp28 \
  --horizon-minutes 10 --hidden-dim 256 --num-layers 3 --dropout 0.2 \
  --epochs 8 --batch-size 512 --learning-rate 8e-4 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp28 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `84.90%`
- Holdout (12 tokens): `88.38%` (samples=5000)

## Notes
- Still below 90%.
