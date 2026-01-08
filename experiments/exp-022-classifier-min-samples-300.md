# Experiment 022 - Classifier filtered to min token samples (300)

## Goal
Improve direction accuracy by training on more established tokens.

## Code changes
- Added `min_token_samples` option to classifier training to filter tokens by candle count.

## Setup
- Model dir: `models_pumpfun_classifier_exp22`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=12, batch_size=512, lr=1e-3
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 300

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp22 \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 12 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp22 \
  --max-tokens 5 --max-samples 2000

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_exp22 \
  --max-tokens 12 --max-samples 5000
```

## Results
- Train validation accuracy: `84.90%`
- Holdout (5 tokens): `90.70%` (samples=2000)
- Holdout (12 tokens): `88.56%` (samples=5000)

## Notes
- This is the first config to clear 90% on a multi-token sample, but it drops below 90% when expanding to 12 tokens.
