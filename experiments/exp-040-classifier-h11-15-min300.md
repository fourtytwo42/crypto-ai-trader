# Experiment 040 - Per-horizon classifiers (11-15m) with min token samples 300

## Goal
Continue per-horizon direction models for 11-15 minutes.

## Setup (shared)
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=1e-3
- Holdout count: 12
- Min token samples: 300
- Backtest: max_tokens=12, max_samples=5000

## Results
- 11m: direction=87.48% (samples=5000)
- 12m: direction=86.96% (samples=5000)
- 13m: direction=86.48% (samples=5000)
- 14m: direction=86.22% (samples=5000)
- 15m: direction=85.74% (samples=5000)

## Notes
- Direction accuracy continues to drop beyond 10m.
