# Experiment 067 - pumpfun multi-horizon (nhits sum20 ctx480 large)

## Goal
Train a single horizon-20 model and evaluate price accuracy for minutes 1..20 on holdout tokens.

## Sync
- skipped (cached training data)

## Config
```json
{
  "model_dir": "experiments/models/exp-067-nhits-sum20-ctx480-large",
  "horizon_minutes": 20,
  "context_length": 480,
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
  "mae": 0.02811248188702979,
  "rmse": 0.052684800721880386,
  "mape": 3.494576603350663
}
```

## Backtest (holdout tokens)
| minutes | price_acc | dir_acc | mae | rmse | smape | samples |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 95.78 | 14.46 | null | null | null | 83 |
| 2 | 95.40 | 12.66 | 0.043431 | 0.064331 | 186.65 | 79 |
| 3 | 94.51 | 6.67 | 0.052143 | 0.088244 | 192.06 | 75 |
| 4 | 93.80 | 9.86 | 0.064175 | 0.113338 | 189.34 | 71 |
| 5 | 91.24 | 10.45 | 0.082864 | 0.121008 | 188.44 | 67 |
| 6 | 89.40 | 12.70 | 0.092032 | 0.147973 | 187.57 | 63 |
| 7 | 86.60 | 13.56 | 0.118313 | 0.181401 | 191.94 | 59 |
| 8 | 83.13 | 10.91 | 0.148752 | 0.212825 | 193.36 | 55 |
| 9 | 73.12 | 11.76 | null | null | null | 51 |
| 10 | 82.15 | 14.58 | 0.166191 | 0.264646 | 191.57 | 48 |
| 11 | 80.77 | 15.56 | 0.170903 | 0.289964 | 187.83 | 45 |
| 12 | 78.18 | 11.90 | 0.194929 | 0.328003 | 193.62 | 42 |
| 13 | 76.31 | 12.82 | 0.217415 | 0.363547 | 194.14 | 39 |
| 14 | 71.45 | 11.11 | null | null | null | 36 |
| 15 | 69.44 | 12.12 | 0.290808 | 0.404708 | 194.64 | 33 |
| 16 | 63.73 | 10.00 | null | null | null | 30 |
| 17 | 63.23 | 11.11 | 0.353648 | 0.469483 | 194.56 | 27 |
| 18 | 64.41 | 12.00 | 0.360351 | 0.499525 | 194.86 | 25 |
| 19 | 65.09 | 13.04 | 0.373620 | 0.527569 | 195.51 | 23 |
| 20 | 63.60 | 9.52 | 0.394943 | 0.562850 | 197.59 | 21 |

## Inference
```json
{
  "avg_seconds": 0.1591147800005274,
  "runs": 5
}
```
