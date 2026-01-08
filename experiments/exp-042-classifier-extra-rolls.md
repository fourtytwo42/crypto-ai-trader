# Experiment 042 - Classifier with additional rolling features

## Goal
Add longer rolling return stats to improve direction accuracy at 10m.

## Code changes
- Added `ret_mean_30`, `ret_std_30`, `ret_mean_120`, `ret_std_120` to classifier feature set.
- Computed these features per token in classifier data prep.

## Setup
- Model dir: `models_pumpfun_classifier_h10_min300_featA`
- Horizon: 10 minutes
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=1e-3
- Holdout count: 12
- Min token samples: 300
- Backtest: max_tokens=12, max_samples=5000

## Commands
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_h10_min300_featA \
  --horizon-minutes 10 --hidden-dim 128 --num-layers 2 --dropout 0.1 \
  --epochs 8 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300

./venv/bin/python -m src.main pumpfun-classify-backtest \
  --model-dir models_pumpfun_classifier_h10_min300_featA \
  --max-tokens 12 --max-samples 5000
```

## Results
- Holdout (10m): direction=89.34% (samples=5000)

## Notes
- Additional rolling features did not cross 90%.
