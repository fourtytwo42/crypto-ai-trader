# Experiment 041 - Per-horizon classifiers (16-20m) with min token samples 300

## Goal
Complete per-horizon direction models for 16-20 minutes.

## Setup (shared)
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=1e-3
- Holdout count: 12
- Min token samples: 300
- Backtest: max_tokens=12, max_samples=5000

## Results
- 16m: direction=85.96% (samples=5000)
- 17m: direction=85.34% (samples=5000)
- 18m: direction=84.96% (samples=5000)
- 19m: direction=84.50% (samples=5000)
- 20m: direction=84.18% (samples=5000)

## Notes
- Direction accuracy declines with horizon length.
