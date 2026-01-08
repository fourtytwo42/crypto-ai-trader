# Experiment 004 - Higher label threshold

## Goal
Improve direction accuracy by filtering low-magnitude returns.

## Setup
- Model dir: `models_pumpfun_classifier_exp04`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=10, batch_size=512, lr=1e-3
- Label threshold: 0.005
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp04 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 10 --batch-size 512 --learning-rate 1e-3 \
  --label-threshold 0.005 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp04 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `82.19%`
- Holdout backtest accuracy: `81.75%` (samples=2000)

## Notes
- Filtering harder reduced accuracy. Not promising.
