# Experiment 037 - Per-horizon classifiers (1-5m) with min token samples 300

## Goal
Train direction classifiers per horizon (1-5 minutes) and evaluate on holdout tokens.

## Setup (shared)
- Architecture: hidden_dim=128, num_layers=2, dropout=0.1
- Training: epochs=8, batch_size=512, lr=1e-3
- Holdout count: 12
- Min token samples: 300
- Backtest: max_tokens=12, max_samples=5000

## Commands (per horizon)
```bash
./venv/bin/python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_h01_min300 --horizon-minutes 1 --hidden-dim 128 --num-layers 2 --dropout 0.1 --epochs 8 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12 --min-token-samples 300
./venv/bin/python -m src.main pumpfun-classify-backtest --model-dir models_pumpfun_classifier_h01_min300 --max-tokens 12 --max-samples 5000

./venv/bin/python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_h02_min300 --horizon-minutes 2 --hidden-dim 128 --num-layers 2 --dropout 0.1 --epochs 8 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12 --min-token-samples 300
./venv/bin/python -m src.main pumpfun-classify-backtest --model-dir models_pumpfun_classifier_h02_min300 --max-tokens 12 --max-samples 5000

./venv/bin/python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_h03_min300 --horizon-minutes 3 --hidden-dim 128 --num-layers 2 --dropout 0.1 --epochs 8 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12 --min-token-samples 300
./venv/bin/python -m src.main pumpfun-classify-backtest --model-dir models_pumpfun_classifier_h03_min300 --max-tokens 12 --max-samples 5000

./venv/bin/python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_h04_min300 --horizon-minutes 4 --hidden-dim 128 --num-layers 2 --dropout 0.1 --epochs 8 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12 --min-token-samples 300
./venv/bin/python -m src.main pumpfun-classify-backtest --model-dir models_pumpfun_classifier_h04_min300 --max-tokens 12 --max-samples 5000

./venv/bin/python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_h05_min300 --horizon-minutes 5 --hidden-dim 128 --num-layers 2 --dropout 0.1 --epochs 8 --batch-size 512 --learning-rate 1e-3 --label-threshold 0.0 --holdout-count 12 --min-token-samples 300
./venv/bin/python -m src.main pumpfun-classify-backtest --model-dir models_pumpfun_classifier_h05_min300 --max-tokens 12 --max-samples 5000
```

## Results
- 1m: direction=95.56% (samples=5000)
- 2m: direction=94.36% (samples=5000)
- 3m: direction=93.30% (samples=5000)
- 4m: direction=92.66% (samples=5000)
- 5m: direction=91.82% (samples=5000)

## Notes
- Direction accuracy stays above 90% for 1-5 minutes using min_token_samples=300.
