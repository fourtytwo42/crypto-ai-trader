# Experiment 066 - pumpfun multi-horizon (nhits sum20 ctx240 mid)

## Goal
Train a single horizon-20 model and evaluate price accuracy for minutes 1..20 on holdout tokens.

## Sync
- skipped (cached training data)

## Config
```json
{
  "model_dir": "experiments/models/exp-066-nhits-sum20-ctx240-mid",
  "horizon_minutes": 20,
  "context_length": 240,
  "model_type": "nhits",
  "target_mode": "sum",
  "hidden_size": 256,
  "num_layers": 2,
  "epochs": 40,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "holdout_count": 12,
  "nhits_stack_types": [
    "identity",
    "identity"
  ],
  "nhits_n_blocks": [
    2,
    1
  ],
  "nhits_mlp_units": [
    [
      256,
      256
    ],
    [
      256,
      256
    ]
  ],
  "nhits_n_pool_kernel_size": [
    2,
    1
  ],
  "nhits_n_freq_downsample": [
    169,
    1
  ]
}
```

## Training Metrics
```json
{
  "mae": 0.03350556997499616,
  "rmse": 0.05671141611809499,
  "mape": 4.374208092134532
}
```

## Backtest (holdout tokens)
| minutes | price_acc | dir_acc | mae | rmse | smape | samples |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96.74 | 2.33 | 0.031751 | 0.050619 | 199.05 | 559 |
| 2 | 92.38 | 4.91 | null | null | null | 550 |
| 3 | 88.31 | 6.84 | 0.125936 | 0.150906 | 197.75 | 541 |
| 4 | 86.61 | 8.83 | 0.148328 | 0.185555 | 196.72 | 532 |
| 5 | 85.70 | 10.90 | 0.159786 | 0.206853 | 195.78 | 523 |
| 6 | 82.53 | 12.62 | 0.200252 | 0.251296 | 195.99 | 515 |
| 7 | 79.46 | 14.20 | 0.240440 | 0.298167 | 196.37 | 507 |
| 8 | 76.56 | 15.23 | 0.282093 | 0.348245 | 196.75 | 499 |
| 9 | 75.35 | 16.90 | 0.299630 | 0.377469 | 196.72 | 491 |
| 10 | 72.44 | 18.22 | 0.343813 | 0.429470 | 196.80 | 483 |
| 11 | 69.94 | 19.37 | 0.384599 | 0.477471 | 197.12 | 475 |
| 12 | 67.78 | 21.20 | 0.419415 | 0.521669 | 196.41 | 467 |
| 13 | 64.92 | 22.00 | 0.471395 | 0.580166 | 196.35 | 459 |
| 14 | 62.49 | 23.06 | 0.513294 | 0.629621 | 196.45 | 451 |
| 15 | 57.76 | 24.38 | 0.604541 | 0.724205 | 196.93 | 443 |
| 16 | 54.25 | 25.29 | 0.677974 | 0.800969 | 197.14 | 435 |
| 17 | 52.49 | 26.93 | 0.722354 | 0.853094 | 196.72 | 427 |
| 18 | 49.38 | 27.92 | 0.795625 | 0.932248 | 196.38 | 419 |
| 19 | 47.44 | 28.95 | 0.847599 | 0.991373 | 196.62 | 411 |
| 20 | 45.74 | 29.78 | 0.898602 | 1.047781 | 196.59 | 403 |

## Inference
```json
{
  "avg_seconds": 0.02517355379968649,
  "runs": 5
}
```
