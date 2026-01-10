# Experiment 069 - pumpfun multi-horizon (nhits sum20 ctx240 default)

## Goal
Train a single horizon-20 model and evaluate price accuracy for minutes 1..20 on holdout tokens.

## Sync
- skipped (cached training data)

## Config
```json
{
  "model_dir": "experiments/models/exp-069-nhits-sum20-ctx240-default",
  "horizon_minutes": 20,
  "context_length": 240,
  "model_type": "nhits",
  "target_mode": "sum",
  "hidden_size": 256,
  "num_layers": 2,
  "epochs": 30,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "holdout_count": 12,
  "nhits_stack_types": null,
  "nhits_n_blocks": null,
  "nhits_mlp_units": null,
  "nhits_n_pool_kernel_size": null,
  "nhits_n_freq_downsample": null
}
```

## Training Metrics
```json
{
  "mae": 0.03125904385838585,
  "rmse": 0.054714751266592976,
  "mape": 4.15916219504097
}
```

## Backtest (holdout tokens)
| minutes | price_acc | dir_acc | mae | rmse | smape | samples |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 97.85 | 2.86 | 0.020410 | 0.040766 | 197.95 | 559 |
| 2 | 96.90 | 3.82 | null | null | null | 550 |
| 3 | 95.65 | 4.81 | 0.040618 | 0.057206 | 197.78 | 541 |
| 4 | 95.32 | 5.64 | 0.044073 | 0.066188 | 196.99 | 532 |
| 5 | 95.05 | 7.07 | 0.047205 | 0.073767 | 195.88 | 523 |
| 6 | 94.29 | 8.93 | 0.054600 | 0.080685 | 195.80 | 515 |
| 7 | 92.74 | 10.85 | null | null | null | 507 |
| 8 | 92.96 | 12.42 | null | null | null | 499 |
| 9 | 92.91 | 11.61 | 0.068653 | 0.098914 | 194.01 | 491 |
| 10 | 91.49 | 11.39 | 0.080795 | 0.110567 | 194.56 | 483 |
| 11 | 88.57 | 9.47 | 0.104528 | 0.137407 | 196.27 | 475 |
| 12 | 85.05 | 9.42 | 0.133539 | 0.169474 | 196.64 | 467 |
| 13 | 82.06 | 8.93 | 0.158153 | 0.194401 | 197.46 | 459 |
| 14 | 77.51 | 8.43 | 0.194246 | 0.230564 | 198.60 | 451 |
| 15 | 72.69 | 8.58 | 0.231317 | 0.267867 | 199.18 | 443 |
| 16 | 65.64 | 8.74 | null | null | null | 435 |
| 17 | 65.06 | 8.90 | 0.287408 | 0.325164 | 199.43 | 427 |
| 18 | 64.92 | 9.07 | 0.288512 | 0.326803 | 199.44 | 419 |
| 19 | 59.90 | 9.25 | 0.324256 | 0.362346 | 199.23 | 411 |
| 20 | 57.71 | 8.93 | 0.340495 | 0.380361 | 199.75 | 403 |

## Inference
```json
{
  "avg_seconds": 0.1073233729985077,
  "runs": 5
}
```
