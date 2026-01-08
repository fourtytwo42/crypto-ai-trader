# Experiment 003 - Wider classifier

## Goal
Increase direction accuracy by expanding model width/depth.

## Setup
- Model dir: `models_pumpfun_classifier_exp03`
- Horizon: 10 minutes
- Architecture: hidden_dim=256, num_layers=3, dropout=0.2
- Training: epochs=10, batch_size=512, lr=5e-4
- Label threshold: 0.0
- Holdout count: 12

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp03 \
  --horizon-minutes 10 --hidden-dim 256 --num-layers 3 --dropout 0.2 \
  --epochs 10 --batch-size 512 --learning-rate 5e-4 \
  --label-threshold 0.0 --holdout-count 12

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp03 \
  --max-tokens 5 --max-samples 2000
```

## Results
- Train validation accuracy: `84.96%`
- Holdout backtest accuracy: `84.6%` (samples=2000)

## Notes
- No improvement over baseline on sampled holdout.
