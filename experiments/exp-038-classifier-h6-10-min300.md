# Experiment 038 - Per-horizon classifiers (6-10m) with min token samples 300

## Goal
Continue per-horizon direction models for 6-10 minutes.

## Setup (shared)
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=1e-3
- Holdout count: 12
- Min token samples: 300
- Backtest: max_tokens=12, max_samples=5000

## Results
- 6m: direction=90.84% (samples=5000)
- 7m: direction=90.40% (samples=5000)
- 8m: direction=89.50% (samples=5000)
- 9m: direction=89.00% (samples=5000)
- 10m: direction=88.64% (samples=5000)

## Notes
- Accuracy drops below 90% from 8m onward with this filter.
