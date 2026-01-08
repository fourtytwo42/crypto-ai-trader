# Experiment 016 - Classifier higher LR, fewer epochs

## Goal
Test faster training with a higher learning rate.

## Setup
- Model dir: `models_pumpfun_classifier_exp16`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=2e-3
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp16 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 8 --batch-size 512 --learning-rate 2e-3 --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp16 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `85.02%`
- Holdout backtest accuracy: `83.50%` (samples=2000)

## Notes
- No improvement.
