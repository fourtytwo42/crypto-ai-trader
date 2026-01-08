# Experiment 039 - Per-horizon classifiers (8-10m) with min token samples 600

## Goal
Increase direction accuracy for 8-10 minutes by filtering to higher-activity tokens.

## Setup (shared)
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=1e-3
- Holdout count: 12
- Min token samples: 600
- Backtest: max_tokens=12, max_samples=5000

## Results
- 8m: direction=89.84% (samples=5000)
- 9m: direction=88.98% (samples=5000)
- 10m: direction=88.50% (samples=5000)

## Notes
- Filtering to 600 samples did not cross 90%.
