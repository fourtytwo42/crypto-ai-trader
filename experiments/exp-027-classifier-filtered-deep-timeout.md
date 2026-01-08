# Experiment 027 - Filtered tokens + deeper model (timeout)

## Goal
Combine min token sample filtering with a deeper classifier.

## Setup
- Model dir: `models_pumpfun_classifier_exp27`
- Horizon: 10 minutes
- Architecture: hidden_dim=256, num_layers=3, dropout=0.2
- Training: epochs=12, batch_size=512, lr=8e-4
- Label threshold: 0.0
- Holdout count: 12
- Min token samples: 300
- Timeout: 180s

## Command
```bash
./venv/bin/python -m src.main pumpfun-classify-train \
  --model-dir models_pumpfun_classifier_exp27 \
  --horizon-minutes 10 --hidden-dim 256 --num-layers 3 --dropout 0.2 \
  --epochs 12 --batch-size 512 --learning-rate 8e-4 --label-threshold 0.0 \
  --holdout-count 12 --min-token-samples 300
```

## Results
- Training timed out at 180s.

## Notes
- Deeper model too slow at this setting.
