# Experiment 065 - pumpfun multi-horizon (nhits sum20 ctx336 large)

## Goal
Train a single horizon-20 model and evaluate price accuracy for minutes 1..20 on holdout tokens.

## Sync
- skipped (cached training data)

## Config
```json
{
  "model_dir": "experiments/models/exp-065-nhits-sum20-ctx336-large",
  "horizon_minutes": 20,
  "context_length": 336,
  "model_type": "nhits",
  "target_mode": "sum",
  "hidden_size": 512,
  "num_layers": 3,
  "epochs": 50,
  "batch_size": 16,
  "learning_rate": 5e-05,
  "holdout_count": 12,
  "nhits_stack_types": [
    "identity",
    "identity",
    "identity"
  ],
  "nhits_n_blocks": [
    3,
    2,
    2
  ],
  "nhits_mlp_units": [
    [
      768,
      768
    ],
    [
      768,
      768
    ],
    [
      768,
      768
    ]
  ],
  "nhits_n_pool_kernel_size": [
    2,
    2,
    1
  ],
  "nhits_n_freq_downsample": [
    4,
    2,
    1
  ]
}
```

## Training Metrics
```json
{
  "mae": 0.028430991371571332,
  "rmse": 0.05263702750541094,
  "mape": 3.100850298460244
}
```

## Backtest (holdout tokens)
| minutes | price_acc | dir_acc | mae | rmse | smape | samples |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.70 | 1.61 | 0.013142 | 0.016754 | 198.16 | 373 |
| 2 | 97.53 | 2.99 | 0.024515 | 0.029914 | 197.71 | 368 |
| 3 | 95.03 | 4.41 | null | null | null | 363 |
| 4 | 96.45 | 4.47 | 0.036227 | 0.051289 | 197.30 | 358 |
| 5 | 95.60 | 5.95 | 0.043680 | 0.055540 | 195.88 | 353 |
| 6 | 95.16 | 7.47 | 0.048225 | 0.061414 | 196.15 | 348 |
| 7 | 95.50 | 9.04 | 0.045811 | 0.065574 | 194.31 | 343 |
| 8 | 94.02 | 7.40 | 0.062078 | 0.082809 | 197.23 | 338 |
| 9 | 93.32 | 8.41 | 0.069554 | 0.089295 | 196.21 | 333 |
| 10 | 90.95 | 9.15 | 0.095192 | 0.110113 | 196.61 | 328 |
| 11 | 86.82 | 9.57 | 0.142138 | 0.155206 | 196.80 | 324 |
| 12 | 83.40 | 9.69 | 0.183143 | 0.196089 | 197.55 | 320 |
| 13 | 79.50 | 11.08 | 0.232041 | 0.245745 | 197.72 | 316 |
| 14 | 74.33 | 11.86 | null | null | null | 312 |
| 15 | 74.45 | 12.66 | 0.299137 | 0.315517 | 197.13 | 308 |
| 16 | 73.85 | 13.49 | 0.307705 | 0.325610 | 196.55 | 304 |
| 17 | 73.34 | 14.00 | 0.314295 | 0.331821 | 197.17 | 300 |
| 18 | 72.86 | 14.53 | 0.321152 | 0.339471 | 196.52 | 296 |
| 19 | 72.50 | 15.75 | null | null | null | 292 |
| 20 | 72.39 | 16.32 | 0.328716 | 0.350612 | 195.58 | 288 |

## Inference
```json
{
  "avg_seconds": 0.12442377879924607,
  "runs": 5
}
```
