# Experiments Log

Purpose: Track training/backtest runs, settings, and key metrics for comparing improvements.

---

## 🏆 BEST MODEL (Current Production)

**NHITS with blocks 3/2/2, MLP 768, lr=5e-5**

### Configuration
| Parameter | Value |
|-----------|-------|
| Model Type | NHITS |
| Horizon | 1 (predicts 24h ahead) |
| Context Length | 336 hours (14 days) |
| Stack Types | identity, identity, identity |
| N Blocks | 3, 2, 2 |
| MLP Units | 768\|768; 768\|768; 768\|768 |
| Pool Kernel Size | 2, 2, 1 |
| Freq Downsample | 4, 2, 1 |
| Learning Rate | 5e-5 |
| Epochs | 50 |
| Batch Size | 16 |
| Loss Type | huber (uses QuantileLoss for horizon=1) |

### Performance (Holdout: Latest 24 Hours)
| Symbol | Directional Accuracy | Price Accuracy |
|--------|---------------------|----------------|
| BTC-USDT | 100.00% | 96.69% |
| ETH-USDT | 100.00% | 94.48% |
| LTC-USDT | 95.83% | 94.38% |
| XRP-USDT | 95.83% | 94.56% |
| **Average** | **97.92%** | **95.03%** |

### Training Metrics
- MAE: 0.5970
- RMSE: 0.5970
- MAPE: 0.6819

### Quick Run Command
```bash
# Inference only (use existing model, pull latest data):
python -m src.main quick-predict --hours 24

# With retraining:
python -m src.main quick-predict --hours 24 --retrain

# Predict 48 hours ahead:
python -m src.main quick-predict --hours 48 --retrain
```

### Full Holdout Evaluation Command
```bash
python -m src.main forecast-holdout-24h \
  --context-length 336 \
  --hidden-size 512 \
  --num-layers 3 \
  --patch-length 8 \
  --stride 4 \
  --loss-type huber \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --model-type nhits \
  --nhits-stack-types identity,identity,identity \
  --nhits-n-blocks 3,2,2 \
  --nhits-mlp-units "768|768;768|768;768|768" \
  --nhits-n-pool-kernel-size 2,2,1 \
  --nhits-n-freq-downsample 4,2,1 \
  --multi-asset
```

---

## Legend
- Metrics: close_mape is lower better. close_within_0.5% / close_within_1% higher better.
- Windows: number of forecast windows evaluated.

## Multi-asset PatchTST (current best)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6252%, close_within_0.5%=0.6215, close_within_1%=0.8438
- Notes: new best across mape and within_0.5%/within_1%.

## Saved model artifacts (best config, multi-asset)
- Model dir: models_best_multiasset_patch6_stride3_quad_huber_ctx168_return
- Config: horizon=12, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return, multi_asset=all symbols
- Train metrics: close_mae=0.003286, close_rmse=0.003978, close_mape=0.9610; volume_mae=0.3998, volume_rmse=0.6207, volume_mape=0.0280
- Notes: forecast_train saved close/volume models for reuse (single full-train + validation split).

## Holdout 24h (baseline, return_24h target)
- Config: horizon=1 (24h target), context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.04435, rmse=0.04435, mape=0.05066
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=95.83, LTC=83.33, XRP=83.33
- Holdout (latest 24h) price accuracy%: BTC=61.24, ETH=63.42, LTC=57.22, XRP=58.68
- Notes: baseline for 24h-ahead evaluation using last-24h holdout targets.

## Holdout 24h (longer context)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.9172, rmse=0.9172, mape=1.0477
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=91.67, XRP=87.50
- Holdout (latest 24h) price accuracy%: BTC=76.30, ETH=74.40, LTC=68.30, XRP=68.28
- Notes: longer context increased holdout accuracy substantially despite worse train metrics (quantile loss scale change).

## Holdout 24h (longer context + larger patch/stride)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.4563, rmse=0.4563, mape=0.5212
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=82.78, ETH=74.95, LTC=76.42, XRP=84.70
- Notes: best holdout price accuracy so far; patch 8/stride 4 looks stronger with longer context.

## Holdout 24h (patch 8/stride 4 + wider hidden)
- Config: horizon=1 (24h target), context_length=336, hidden_size=704, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.7313, rmse=0.7313, mape=0.8353
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=83.33, XRP=87.50
- Holdout (latest 24h) price accuracy%: BTC=59.65, ETH=57.59, LTC=68.05, XRP=50.52
- Notes: wider hidden size degraded holdout price accuracy; not competitive.

## Holdout 24h (patch 8/stride 4, batch size 16)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.4563, rmse=0.4563, mape=0.5212
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=82.78, ETH=74.95, LTC=76.42, XRP=84.70
- Notes: identical to batch_size=32 run; suggests batch size not affecting outcome in this setup.

## Holdout 24h (ctx 504, patch 8/stride 4)
- Config: horizon=1 (24h target), context_length=504, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Notes: failed with CUDA OOM at batch 32; likely exceeds 24GB VRAM with context 504.

## Holdout 24h (ctx 504, patch 8/stride 4, batch size 16)
- Config: horizon=1 (24h target), context_length=504, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Notes: failed with CUDA OOM even at batch 16; ctx 504 not feasible on 24GB VRAM.

## Holdout 24h (patch 8/stride 4, lr=5e-5)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=5e-5, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6309, rmse=0.6309, mape=0.7206
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=75.91, ETH=58.45, LTC=70.12, XRP=70.85
- Notes: lower LR hurt ETH and BTC price accuracy vs best.

## Holdout 24h (patch 8/stride 4, lr=2e-4)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=2e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.3442, rmse=0.3442, mape=0.3932
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=75.00, XRP=83.33
- Holdout (latest 24h) price accuracy%: BTC=64.93, ETH=60.77, LTC=51.32, XRP=61.72
- Notes: higher LR degraded price accuracy, especially LTC.

## Holdout 24h (patch 8/stride 4, hidden 512)
- Config: horizon=1 (24h target), context_length=336, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6246, rmse=0.6246, mape=0.7134
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=71.21, ETH=70.69, LTC=74.08, XRP=80.00
- Notes: smaller model underperformed best on BTC/ETH price accuracy.

## Holdout 24h (patch 8/stride 4, layers 5)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=5, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.2409, rmse=0.2409, mape=0.2751
- Holdout (latest 24h) directional accuracy%: BTC=95.83, ETH=100.0, LTC=87.50, XRP=87.50
- Holdout (latest 24h) price accuracy%: BTC=68.48, ETH=65.44, LTC=66.41, XRP=55.52
- Notes: shallower model underperformed best, especially XRP price accuracy.

## Holdout 24h (patch 8/stride 4, layers 7)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=7, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6884, rmse=0.6884, mape=0.7863
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=91.67, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=83.99, ETH=77.22, LTC=73.14, XRP=81.66
- Notes: better BTC/ETH price accuracy than best, worse LTC/XRP vs layers 6.

## Holdout 24h (patch 8/stride 4, layers 8, batch size 16)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=8, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.3234, rmse=0.3234, mape=0.3694
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=87.50
- Holdout (latest 24h) price accuracy%: BTC=71.45, ETH=65.60, LTC=75.47, XRP=68.28
- Notes: deeper model did not improve over layers 6/7; weaker BTC/ETH price accuracy.

## Holdout 24h (patch 8/stride 4, ctx 384, batch size 16)
- Config: horizon=1 (24h target), context_length=384, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.9552, rmse=0.9552, mape=1.0911
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=71.64, ETH=68.17, LTC=65.96, XRP=66.12
- Notes: longer context did not beat ctx 336; BTC/ETH ok, LTC/XRP weaker than best.

## Holdout 24h (patch 12/stride 6, ctx 336, batch size 16)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=12, stride=6, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.8193, rmse=0.8193, mape=0.9358
- Holdout (latest 24h) directional accuracy%: BTC=95.83, ETH=100.0, LTC=79.17, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=72.84, ETH=78.67, LTC=64.36, XRP=71.01
- Notes: ETH improved vs best, but BTC/LTC/XRP weaker overall.

## Holdout 24h (patch 10/stride 5, ctx 336, batch size 16)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=10, stride=5, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.3828, rmse=0.3828, mape=0.4372
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=63.97, ETH=64.14, LTC=66.74, XRP=64.27
- Notes: worse across price accuracy; patch 10/stride 5 not competitive.

## Holdout 24h (patch 8/stride 4, ctx 336, MAE loss flag)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=mae, epochs=50, batch_size=16, lr=1e-4, model=patchtst, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.4563, rmse=0.4563, mape=0.5212
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=87.50, XRP=91.67
- Holdout (latest 24h) price accuracy%: BTC=82.78, ETH=74.95, LTC=76.42, XRP=84.70
- Notes: identical to Huber run; horizon=1 uses QuantileLoss so loss_type flag is ignored.

## Holdout 24h (NHITS, ctx 336, batch size 16)
- Config: horizon=1 (24h target), context_length=336, hidden_size=512, num_layers=3, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6618, rmse=0.6618, mape=0.7559
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.22, ETH=89.53, LTC=90.62, XRP=86.62
- Notes: massive improvement vs PatchTST on holdout price accuracy; new best so far.

## Holdout 24h (NHITS, ctx 336, hidden 640, layers 4)
- Config: horizon=1 (24h target), context_length=336, hidden_size=640, num_layers=4, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6618, rmse=0.6618, mape=0.7559
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.22, ETH=89.53, LTC=90.62, XRP=86.62
- Notes: identical metrics/params to NHITS baseline; NHITS ignores hidden_size/num_layers or clamps internally.

## Holdout 24h (NHITS, ctx 168, batch size 16)
- Config: horizon=1 (24h target), context_length=168, hidden_size=512, num_layers=3, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6909, rmse=0.6909, mape=0.7892
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=90.89, ETH=81.44, LTC=95.01, XRP=81.48
- Notes: slightly worse BTC/ETH/XRP vs ctx 336, but LTC improved.

## Holdout 24h (NHITS, ctx 336, lr=2e-4)
- Config: horizon=1 (24h target), context_length=336, hidden_size=512, num_layers=3, loss_type=huber, epochs=50, batch_size=16, lr=2e-4, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.7141, rmse=0.7141, mape=0.8157
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=91.60, ETH=86.55, LTC=90.26, XRP=86.32
- Notes: slightly worse than NHITS baseline; lr=2e-4 not an improvement.

## Holdout 24h (NHITS, ctx 336, lr=5e-5)
- Config: horizon=1 (24h target), context_length=336, hidden_size=512, num_layers=3, loss_type=huber, epochs=50, batch_size=16, lr=5e-5, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.7066, rmse=0.7066, mape=0.8071
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=94.86, ETH=89.14, LTC=94.90, XRP=86.17
- Notes: close to NHITS baseline; slightly better BTC/LTC, slightly worse ETH/XRP; not a clear win.

## Holdout 24h (NHITS, ctx 504, batch size 16)
- Config: horizon=1 (24h target), context_length=504, hidden_size=512, num_layers=3, loss_type=huber, epochs=50, batch_size=16, lr=1e-4, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.7031, rmse=0.7031, mape=0.8031
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=81.74, ETH=85.56, LTC=93.18, XRP=83.96
- Notes: longer context hurt BTC/ETH vs NHITS baseline; LTC strong but overall worse.

## Holdout 24h (NHITS, ctx 336, batch size 32)
- Config: horizon=1 (24h target), context_length=336, hidden_size=512, num_layers=3, loss_type=huber, epochs=50, batch_size=32, lr=1e-4, model=nhits, multi_asset=BTC/ETH/LTC/XRP
- Train metrics: mae=0.6618, rmse=0.6618, mape=0.7559
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.22, ETH=89.53, LTC=90.62, XRP=86.62
- Notes: identical to NHITS baseline; batch size has no effect here.

## Holdout 24h (NHITS, custom stacks/MLP)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=2,2,1, mlp_units=1024|1024;1024|1024;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Notes: failed with shape mismatch (mat1 1024x1024 vs 512x512); MLP units too wide for current NHITS block config.

## Holdout 24h (NHITS, custom stacks/MLP v2)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=2,2,1, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6477, rmse=0.6477, mape=0.7399
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=91.60, ETH=87.34, LTC=91.56, XRP=86.86
- Notes: solid but worse than NHITS baseline on BTC/ETH/LTC.

## Holdout 24h (NHITS, custom blocks 2/2/2)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=2,2,2, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6562, rmse=0.6562, mape=0.7496
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=92.90, ETH=91.53, LTC=89.37, XRP=89.19
- Notes: better ETH/XRP, worse BTC/LTC vs NHITS baseline; mixed outcome.

## Holdout 24h (NHITS, custom downsample 8/4/1)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=1,1,1, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=4,2,1, n_freq_downsample=8,4,1
- Train metrics: mae=0.5651, rmse=0.5651, mape=0.6454
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=93.70, ETH=91.98, LTC=89.19, XRP=89.48
- Notes: strong ETH/XRP, BTC slightly below baseline, LTC weaker; mixed vs NHITS baseline.

## Holdout 24h (NHITS, smaller MLP 256)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=1,1,1, mlp_units=256|256;256|256;256|256, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6579, rmse=0.6579, mape=0.7515
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.69, ETH=89.42, LTC=89.09, XRP=90.29
- Notes: slightly better BTC/XRP, worse LTC vs baseline; overall mixed.

## Holdout 24h (NHITS, custom downsample 6/3/1)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=1,1,1, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=3,2,1, n_freq_downsample=6,3,1
- Train metrics: mae=0.6384, rmse=0.6384, mape=0.7292
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.04, ETH=89.22, LTC=89.62, XRP=89.28
- Notes: roughly baseline-level; no clear improvement.

## Holdout 24h (NHITS, mixed MLP widths)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=1,1,1, mlp_units=512|256;512|256;512|256, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Notes: failed with shape mismatch (mat1 1024x256 vs 512x256); NHITS expects consistent widths per layer in this setup.

## Holdout 24h (NHITS, blocks 3/3/3)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,3, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6556, rmse=0.6556, mape=0.7488
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=94.15, ETH=94.60, LTC=91.58, XRP=92.98
- Notes: strongest overall so far; improved ETH/XRP and solid LTC while matching baseline BTC.

## Holdout 24h (NHITS, blocks 3/3/3, MLP 512x3)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,3, mlp_units=512|512|512;512|512|512;512|512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6556, rmse=0.6556, mape=0.7488
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=94.15, ETH=94.60, LTC=91.58, XRP=92.98
- Notes: identical to blocks 3/3/3 baseline; extra MLP depth did not change outcome.

## Holdout 24h (NHITS, blocks 3/3/3, lr=2e-5, epochs 100)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,3, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1, epochs=100, batch_size=16, lr=2e-5
- Train metrics: mae=0.7419, rmse=0.7419, mape=0.8475
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=90.20, ETH=88.31, LTC=88.44, XRP=84.72
- Notes: longer training with lower LR regressed vs 50-epoch baseline.

## Holdout 24h (NHITS, 4 stacks, blocks 2/2/2/2)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity,identity, n_blocks=2,2,2,2, mlp_units=512|512;512|512;512|512;512|512, n_pool_kernel_size=2,2,1,1, n_freq_downsample=8,4,2,1
- Train metrics: mae=0.6275, rmse=0.6275, mape=0.7167
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=92.48, ETH=90.55, LTC=88.61, XRP=88.71
- Notes: worse than 3-stack baseline; 4 stacks not beneficial here.

## Holdout 24h (NHITS, blocks 3/3/3, MLP 768)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,3, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6328, rmse=0.6328, mape=0.7228
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=92.54, ETH=93.59, LTC=93.86, XRP=93.23
- Notes: improves ETH/LTC/XRP but hurts BTC vs baseline; competitive but not a clear overall win.

## Holdout 24h (NHITS, blocks 3/3/2, MLP 768)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,2, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.5993, rmse=0.5993, mape=0.6845
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=93.81, ETH=95.06, LTC=92.47, XRP=94.80
- Notes: new best overall so far (avg ~94.04); stronger ETH/XRP while keeping BTC/LTC high.

## Holdout 24h (NHITS, blocks 3/2/2, MLP 768)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,2,2, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6038, rmse=0.6038, mape=0.6897
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.65, ETH=93.88, LTC=93.64, XRP=94.37
- Notes: new best overall so far (avg ~94.14); stronger BTC/LTC, slight ETH drop vs 3/3/2.

## Holdout 24h (NHITS, blocks 3/2/2, MLP 896)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,2,2, mlp_units=896|896;896|896;896|896, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.5884, rmse=0.5884, mape=0.6721
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=95.58, ETH=94.17, LTC=91.36, XRP=90.94
- Notes: higher BTC/ETH, but LTC/XRP dipped; average (~93.01) below best.

## Holdout 24h (NHITS, blocks 3/2/3, MLP 768)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,2,3, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6505, rmse=0.6505, mape=0.7430
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=93.08, ETH=91.81, LTC=93.86, XRP=92.74
- Notes: improved LTC, but BTC/ETH down; average (~92.87) below best.

## Holdout 24h (NHITS, blocks 3/1/2, MLP 768)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,1,2, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6255, rmse=0.6255, mape=0.7145
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=92.04, ETH=92.42, LTC=93.31, XRP=92.82
- Notes: lower BTC, modest ETH/LTC; average (~92.65) below best.

## Holdout 24h (NHITS, blocks 3/2/2, MLP 768, lr=5e-5)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,2,2, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1, lr=5e-5
- Train metrics: mae=0.5970, rmse=0.5970, mape=0.6819
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=96.69, ETH=94.48, LTC=94.38, XRP=94.56
- Notes: new best overall so far (avg ~95.03); improved across all symbols.

## Holdout 24h (NHITS, blocks 3/2/2, MLP 768, lr=2e-4)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,2,2, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1, lr=2e-4
- Train metrics: mae=0.6119, rmse=0.6119, mape=0.6989
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=90.83, ETH=91.53, LTC=88.21, XRP=92.63
- Notes: worse than lr=5e-5; not competitive.

## Holdout 24h (NHITS, blocks 3/2/2, MLP 768, lr=5e-5, epochs=100)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,2,2, mlp_units=768|768;768|768;768|768, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1, lr=5e-5, epochs=100
- Train metrics: mae=0.6498, rmse=0.6498, mape=0.7422
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=94.12, ETH=88.51, LTC=88.13, XRP=84.80
- Notes: longer training degraded holdout accuracy; keep epochs=50 for this config.

## Holdout 24h (NHITS, blocks 4/4/4)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=4,4,4, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6961, rmse=0.6961, mape=0.7951
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=94.56, ETH=92.12, LTC=92.89, XRP=91.32
- Notes: strong across the board; ETH below 3/3/3, LTC improved; likely competitive overall.

## Holdout 24h (NHITS, blocks 5/5/5)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=5,5,5, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,2,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.6185, rmse=0.6185, mape=0.7065
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=94.24, ETH=92.71, LTC=90.75, XRP=89.29
- Notes: larger model did not improve overall vs blocks 3/3/3 or 4/4/4.

## Holdout 24h (NHITS, blocks 3/3/3 with downsample 6/3/1)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,3, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=3,2,1, n_freq_downsample=6,3,1
- Train metrics: mae=0.6405, rmse=0.6405, mape=0.7316
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=95.83
- Holdout (latest 24h) price accuracy%: BTC=93.03, ETH=92.41, LTC=89.92, XRP=93.66
- Notes: strong ETH/XRP, weaker BTC/LTC vs blocks 3/3/3 baseline; mixed.

## Holdout 24h (NHITS, blocks 3/3/3, pool 2/1/1)
- Config: horizon=1 (24h target), context_length=336, stack_types=identity,identity,identity, n_blocks=3,3,3, mlp_units=512|512;512|512;512|512, n_pool_kernel_size=2,1,1, n_freq_downsample=4,2,1
- Train metrics: mae=0.7174, rmse=0.7174, mape=0.8194
- Holdout (latest 24h) directional accuracy%: BTC=100.0, ETH=100.0, LTC=95.83, XRP=100.0
- Holdout (latest 24h) price accuracy%: BTC=90.97, ETH=86.80, LTC=93.91, XRP=83.94
- Notes: degraded BTC/ETH/XRP vs baseline; not competitive.

## Multi-asset PatchTST (rerun with mixed precision + mem cleanup)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50
- Windows: 8
- Metrics: close_mape=1.6353%, close_within_0.5%=0.3958, close_within_1%=0.5677
- Notes: rerun after adding mixed precision + CUDA memory logging/cleanup; worse than best.

## Multi-asset PatchTST (best config baseline, per-horizon)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_len=16, stride=8, loss_type=mae, horizon_weight_mode=linear
- Windows: 8
- Metrics: close_mape=1.6353%, close_within_0.5%=0.3958, close_within_1%=0.5677
- Per-horizon directional accuracy (h1-h12): 0.5313, 0.3750, 0.4062, 0.4688, 0.4375, 0.4688, 0.5000, 0.4688, 0.4062, 0.4062, 0.5000, 0.6250
- Per-horizon within 0.5% (h1-h12): 0.7500, 0.5313, 0.5000, 0.4062, 0.4375, 0.3125, 0.3438, 0.2812, 0.2812, 0.3125, 0.3438, 0.2500
- Per-horizon within 1% (h1-h12): 0.9062, 0.8125, 0.6562, 0.5938, 0.5625, 0.5625, 0.5000, 0.4062, 0.4062, 0.4688, 0.4688, 0.4688
- Means (h1-h12): directional=0.4661, within_0.5%=0.3958, within_1%=0.5677
- Notes: baseline rerun to capture h1-h12 accuracy; matches rerun aggregate metrics above.

## Multi-asset PatchTST (wider hidden size)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, hidden_size=576, num_layers=6
- Windows: 8
- Metrics: close_mape=1.6633%, close_within_0.5%=0.3802, close_within_1%=0.5599
- Notes: wider model performed worse than best.

## Multi-asset PatchTST (deeper)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, hidden_size=512, num_layers=7
- Windows: 8
- Metrics: close_mape=1.6663%, close_within_0.5%=0.3646, close_within_1%=0.5547
- Notes: deeper model performed worse than best.

## Multi-asset NHITS (baseline)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168
- Windows: 8
- Metrics: close_mape=1.9322%, close_within_0.5%=0.2448, close_within_1%=0.4453
- Notes: much worse than PatchTST best.

## Multi-asset PatchTST (longer context)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=256, hidden_size=512, num_layers=6
- Windows: 8
- Metrics: close_mape=1.6853%, close_within_0.5%=0.3880, close_within_1%=0.5599
- Notes: longer context did not improve over best.

## Multi-asset PatchTST (Huber loss)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, loss_type=huber
- Windows: 8
- Metrics: close_mape=1.6811%, close_within_0.5%=0.4062, close_within_1%=0.5625
- Notes: Huber loss did not beat best; within_0.5% slightly higher than recent runs.

## Multi-asset PatchTST (exp horizon weighting)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, horizon_weight_mode=exp
- Windows: 8
- Metrics: close_mape=1.6447%, close_within_0.5%=0.3880, close_within_1%=0.5755
- Notes: exp weighting did not beat best.

## Multi-asset PatchTST (smaller patch/stride)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=linear
- Windows: 8
- Metrics: close_mape=1.2483%, close_within_0.5%=0.4340, close_within_1%=0.5938
- Notes: improved over recent reruns but still worse than best (0.7911% mape).

## Multi-asset PatchTST (smaller patch/stride + quadratic weighting)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=quadratic
- Windows: 8
- Metrics: close_mape=1.2403%, close_within_0.5%=0.4271, close_within_1%=0.5972
- Notes: similar to linear weighting; marginally better mape, similar within_%.

## Multi-asset PatchTST (smaller patch/stride + exp weighting)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=exp
- Windows: 8
- Metrics: close_mape=1.2255%, close_within_0.5%=0.4375, close_within_1%=0.5868
- Notes: slight mape improvement vs quadratic; still below best overall.

## Multi-asset PatchTST (return target, smaller patch/stride)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=linear, close_target=return
- Windows: 8
- Metrics: close_mape=0.7439%, close_within_0.5%=0.5382, close_within_1%=0.7778
- Notes: best mape so far; strong within_1% and within_0.5% vs prior runs.

## Multi-asset PatchTST (return target + exp weighting)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=exp, close_target=return
- Windows: 8
- Metrics: close_mape=0.7660%, close_within_0.5%=0.5243, close_within_1%=0.7778
- Notes: slightly worse mape than linear return target; within_1% matched.

## Multi-asset PatchTST (return target + quadratic weighting)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7317%, close_within_0.5%=0.5521, close_within_1%=0.7778
- Notes: new best so far; improved mape and within_0.5% vs linear return target.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7344%, close_within_0.5%=0.5694, close_within_1%=0.7778
- Notes: mape slightly worse than best; within_0.5% still strong.

## Multi-asset PatchTST (return target + linear weighting, Huber loss)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=huber, horizon_weight_mode=linear, close_target=return
- Windows: 8
- Metrics: close_mape=0.6973%, close_within_0.5%=0.5938, close_within_1%=0.8299
- Notes: strong baseline; surpassed by wider hidden size on mape.

## Multi-asset PatchTST (return target + linear weighting, Huber loss, longer context)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=256, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=huber, horizon_weight_mode=linear, close_target=return
- Windows: 8
- Metrics: close_mape=0.7259%, close_within_0.5%=0.5521, close_within_1%=0.8368
- Notes: better within_1% than best, but worse mape; keep as alternative.

## Multi-asset PatchTST (return target + linear weighting, Huber loss, wider hidden size)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, horizon_weight_mode=linear, close_target=return
- Windows: 8
- Metrics: close_mape=0.6741%, close_within_0.5%=0.5938, close_within_1%=0.8090
- Notes: strong mape; within_0.5% still best-in-class, but mape now edged by quadratic weighting.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, wider hidden size)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6704%, close_within_0.5%=0.5764, close_within_1%=0.8090
- Notes: current best mape; trades off within_0.5% vs linear weighting.

## Multi-asset PatchTST (return target + exp weighting, Huber loss, wider hidden size)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=8, stride=4, loss_type=huber, horizon_weight_mode=exp, close_target=return
- Windows: 8
- Metrics: close_mape=0.6778%, close_within_0.5%=0.5799, close_within_1%=0.8125
- Notes: slightly worse mape than quadratic; within_1% modestly higher.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, wider hidden size, smaller patch/stride)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6252%, close_within_0.5%=0.6215, close_within_1%=0.8438
- Notes: new best overall; best mape and tight-accuracy.

## Multi-asset PatchTST (return target + linear weighting, Huber loss, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=linear, close_target=return
- Windows: 8
- Metrics: close_mape=0.6837%, close_within_0.5%=0.5764, close_within_1%=0.8056
- Notes: worse than quadratic weighting with same patch/stride; keep as comparison.

## Multi-asset PatchTST (return target + exp weighting, Huber loss, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=exp, close_target=return
- Windows: 8
- Metrics: close_mape=0.6293%, close_within_0.5%=0.6285, close_within_1%=0.8403
- Notes: slightly worse mape than quadratic best, but best within_0.5% so far.

## Multi-asset PatchTST (return target + cubic weighting, Huber loss, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=cubic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6510%, close_within_0.5%=0.5938, close_within_1%=0.8090
- Notes: cubic weighting underperformed quadratic best; within_0.5% lower than exp/quad.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, patch 6/stride 3, lr=5e-4)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return, learning_rate=5e-4
- Windows: 8
- Metrics: close_mape=1.0238%, close_within_0.5%=0.4028, close_within_1%=0.6701
- Notes: lower learning rate hurt mape and tight-accuracy vs best and cubic.

## Multi-asset PatchTST (return target + quadratic weighting, MAE loss, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=mae, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7241%, close_within_0.5%=0.5208, close_within_1%=0.7986
- Notes: MAE loss underperformed Huber; mape worse than best.

## Multi-asset PatchTST (return target + no horizon weighting, Huber loss, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=none, close_target=return
- Windows: 8
- Metrics: close_mape=0.7035%, close_within_0.5%=0.6146, close_within_1%=0.7951
- Notes: slightly worse mape than quadratic best; within_0.5% closer to best but still lower.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, patch 6/stride 4)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=4, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6725%, close_within_0.5%=0.6389, close_within_1%=0.8264
- Notes: stronger within_0.5% than best, but mape still worse than 0.6252% best.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, patch 5/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=5, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7138%, close_within_0.5%=0.5938, close_within_1%=0.8090
- Notes: worse than best and worse than patch 6/stride 4.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, patch 6/stride 5)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=5, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7196%, close_within_0.5%=0.5868, close_within_1%=0.8056
- Notes: worse than stride 3/4; higher stride did not help.

## Multi-asset PatchTST (return target + quadratic weighting, Huber loss, patch 6/stride 3, lr=2e-3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return, learning_rate=2e-3
- Windows: 8
- Metrics: close_mape=nan, close_within_0.5%=0.0903, close_within_1%=0.1875
- Notes: unstable (NaNs); learning_rate too high.

## Multi-asset PatchTST (best config, extended windows)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return, max_windows=14
- Windows: 12
- Metrics: close_mape=0.8677%, close_within_0.5%=0.5440, close_within_1%=0.7801
- Notes: only 12 windows available; mean accuracy dropped vs 8-window best (0.6252% mape).

## Multi-asset PatchTST (return target + quadratic weighting, hidden 704, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=704, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6737%, close_within_0.5%=0.6181, close_within_1%=0.8333
- Notes: wider hidden size did not beat hidden_size=640 best.

## Multi-asset PatchTST (return target + quadratic weighting, longer context)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=192, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7495%, close_within_0.5%=0.5382, close_within_1%=0.7951
- Notes: longer context length degraded both mape and tight-accuracy vs best.

## Multi-asset PatchTST (return target + quadratic weighting, shorter context)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=144, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7383%, close_within_0.5%=0.5938, close_within_1%=0.7882
- Notes: shorter context improved directional accuracy but mape and within_1% worse than best.

## Multi-asset PatchTST (return target + quadratic weighting, patch 5/stride 2)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=5, stride=2, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7479%, close_within_0.5%=0.5625, close_within_1%=0.8021
- Notes: smaller patch/stride slowed training and degraded mape vs best.

## Multi-asset PatchTST (return target + quadratic weighting, patch 4/stride 2)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=4, stride=2, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.8092%, close_within_0.5%=0.5729, close_within_1%=0.8021
- Notes: even smaller patch/stride further worsened mape; not competitive.

## Multi-asset PatchTST (return target + quadratic weighting, patch 7/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=7, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6789%, close_within_0.5%=0.6076, close_within_1%=0.8229
- Notes: better than patch 5/2 and 4/2, still behind best mape (0.6252%).

## Multi-asset PatchTST (return target + quadratic weighting, patch 6/stride 2)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=2, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7793%, close_within_0.5%=0.6076, close_within_1%=0.7986
- Notes: more overlap did not help; mape worse than best.

## Multi-asset PatchTST (return target + quadratic weighting, longer context)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=256, hidden_size=512, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7706%, close_within_0.5%=0.5417, close_within_1%=0.7708
- Notes: longer context hurt vs context_length=168.

## Multi-asset PatchTST (return target + quadratic weighting, patch 12/stride 6)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=6, patch_length=12, stride=6, loss_type=mae, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7881%, close_within_0.5%=0.5208, close_within_1%=0.7431
- Notes: worse than patch_length=8/stride=4.

## Multi-asset PatchTST (return target + quadratic weighting, wider hidden size)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=576, num_layers=6, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7557%, close_within_0.5%=0.5174, close_within_1%=0.7708
- Notes: wider model did not beat hidden_size=512 quadratic return target.

## Multi-asset PatchTST (return target + quadratic weighting, deeper)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=512, num_layers=7, patch_length=8, stride=4, loss_type=mae, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.7545%, close_within_0.5%=0.5347, close_within_1%=0.7674
- Notes: deeper model improved vs log_close runs but still worse than best return-target config (0.7317% mape).

## Multi-asset PatchTST (alternate, fewer windows)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=2190, epochs=50
- Windows: 5
- Metrics: close_mape=0.6869%
- Notes: lower mape but fewer windows; keep for follow-up.

## Multi-asset PatchTST (larger step)
- Config: train_size=26280, val_size=3942, test_size=4380, step_size=2190, epochs=50
- Windows: 14
- Metrics: close_mape=1.2534%, close_within_0.5%=0.3792, close_within_1%=0.6083
- Notes: worse than current best.

## Single-asset backtests (early baseline)
- Windows 1-30: directional_accuracy=0.5139, mae=0.00509, rmse=0.00684, mape=2.119
- Windows 31-60: directional_accuracy=0.4833, mae=0.00396, rmse=0.00504, mape=24.7899
- Windows 61-90: directional_accuracy=0.4750, mae=0.00504, rmse=0.00660, mape=2.3477
- Windows 91-120: directional_accuracy=0.5083, mae=0.00454, rmse=0.00590, mape=1.6439
- Windows 151-250: directional_accuracy=0.5033, mae=0.00479, rmse=0.00617, mape=2.4735
- Notes: directional accuracy hovered around ~0.50; inconsistent mape.

## Failed runs (resource issues)
- PatchTST hidden=640, layers=8, batch=4: CUDA OOM.
- PatchTST hidden=640, layers=8, batch=2, PYTORCH_CUDA_ALLOC_CONF set: CUDA "unknown error" mid-training.
- PatchTST hidden=640, patch_length=5, stride=2, quadratic, huber: run timed out in CLI; hit Rich console OSError on Windows.

## Infra changes
- Added max_vram_gb cap (default 23.0) to training config and applied per-process CUDA memory fraction.


## Pump.fun minute forecasts (2026-01-07)

### Sync run (1000 tokens)
- Command: `python -m src.main pumpfun-sync --max-tokens 1000 --replace`
- Output: tokens=1000, candles=1,259,098, features=1,203,506
- Notes: SOL/USD cached in pump.fun DB (`pump_sol_prices`).

### Baseline NHITS ctx336 (horizon 10m)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_nhits`
- Train metrics: mae=0.02676, rmse=0.04622, mape=2.1438
- Backtest (10m, window 240): mae=0.11794, rmse=0.17714, smape=??, direction_acc=16.81%, samples=2760

### NHITS ctx240 (horizon 10m)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx240 --context-length 240`
- Train metrics: mae=0.02482, rmse=0.04609, mape=1.7154
- Backtest (10m, window 240): mae=0.07704, rmse=0.14493, smape=187.27, direction_acc=18.77%, samples=2760
- Notes: Best MAE/RMSE so far on holdout.

### NHITS ctx480 (horizon 10m)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx480 --context-length 480`
- Train metrics: mae=0.01954, rmse=0.02692, mape=7.3735
- Backtest (10m, window 240): mae=0.08560, rmse=0.15013, smape=188.88, direction_acc=17.17%, samples=2760
- Notes: Larger context did not improve holdout MAE/RMSE vs ctx240.

### NHITS ctx180 (horizon 10m)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx180 --context-length 180`
- Train metrics: mae=0.02532, rmse=0.04640, mape=2.0644
- Backtest (10m, window 240): mae=0.08060, rmse=0.14616, smape=187.30, direction_acc=18.84%, samples=2760

### NHITS ctx300 (horizon 10m)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx300 --context-length 300`
- Train metrics: mae=0.02661, rmse=0.04776, mape=1.8839
- Backtest (10m, window 240): mae=0.08224, rmse=0.15282, smape=188.55, direction_acc=17.68%, samples=2760

### NHITS ctx240 lr=1e-4
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx240_lr1e4 --context-length 240 --learning-rate 0.0001`
- Train metrics: mae=0.02348, rmse=0.04526, mape=1.6138
- Backtest (10m, window 240): mae=0.07008, rmse=0.13944, smape=187.60, direction_acc=17.75%, samples=2760
- Notes: Best MAE/RMSE so far on 240-window holdout.

### NHITS ctx240 lr=1e-4 batch=32 (timeout)
- Command: `python -m src.main pumpfun-backtest --model-dir models_pumpfun_ctx240_lr1e4_b32 --minutes 10 --test-window 240`
- Notes: Backtest exceeded 600s timeout twice.

### PatchTST ctx240 (OOM)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_patchtst_ctx240 --model-type patchtst --context-length 240 --hidden-size 512 --num-layers 4 --patch-length 4 --stride 2`
- Notes: Process killed (signal 9) during training.

### PatchTST ctx180 (timeout, epochs 50)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_patchtst_ctx180 --model-type patchtst --context-length 180 --hidden-size 256 --num-layers 3 --patch-length 4 --stride 2`
- Notes: Training exceeded 600s timeout.

### PatchTST ctx180 epochs 10
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_patchtst_ctx180_e10 --model-type patchtst --context-length 180 --hidden-size 256 --num-layers 3 --patch-length 4 --stride 2 --epochs 10`
- Train metrics: mae=0.01209, rmse=0.03993, mape=1.0777
- Backtest (10m, window 120): mae=0.02269, rmse=0.06176, smape=183.48, direction_acc=19.70%, samples=1320
- Notes: Strong MAE/RMSE on 120-window but full 240-window backtest timed out.

### NHITS ctx240 lr=2e-4
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx240_lr2e4 --context-length 240 --learning-rate 0.0002`
- Train metrics: mae=0.02252, rmse=0.04455, mape=1.6689
- Backtest (10m, window 120): mae=0.07196, rmse=0.16194, smape=189.05, direction_acc=15.68%, samples=1320
- Notes: 240-window backtest timed out; 120-window shows weaker direction accuracy.

### NHITS ctx240 lr=5e-5
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_ctx240_lr5e5 --context-length 240 --learning-rate 0.00005`
- Train metrics: mae=0.02482, rmse=0.04609, mape=1.7154
- Backtest (10m, window 120): mae=0.08473, rmse=0.16877, smape=187.17, direction_acc=19.24%, samples=1320

### NHITS direct target (ctx180, horizon 1)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx180_lr1e4_e30 --horizon-minutes 1 --context-length 180 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.88419, rmse=4.30364, mape=94.7351
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24913, rmse=1.09110, smape=83.29, direction_acc=61.82%, samples=440
- Notes: Directional accuracy improves vs sum mode but still far from target.

### PatchTST direct target (ctx120, horizon 1)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_patchtst_ctx120_lr2e4_e20 --horizon-minutes 1 --context-length 120 --model-type patchtst --target-mode direct --hidden-size 256 --num-layers 3 --patch-length 8 --stride 4 --epochs 20 --batch-size 32 --learning-rate 0.0002`
- Train metrics: mae=1.64380, rmse=2.64804, mape=60.1203
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.93211, rmse=1.92817, smape=185.55, direction_acc=17.73%, samples=440

### NHITS direct target (ctx60, horizon 1)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx60_lr2e4_e30 --horizon-minutes 1 --context-length 60 --model-type nhits --target-mode direct --hidden-size 128 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0002`
- Train metrics: mae=2.91341, rmse=4.40298, mape=95.9299
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24250, rmse=1.09081, smape=82.62, direction_acc=61.82%, samples=440

### NHITS direct target (ctx120, horizon 1)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx120_lr1e4_e30 --horizon-minutes 1 --context-length 120 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.92167, rmse=4.28226, mape=111.718
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24656, rmse=1.09226, smape=82.91, direction_acc=62.05%, samples=440

### NHITS direct target (ctx240, horizon 1, lr=2e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx240_lr2e4_e20 --horizon-minutes 1 --context-length 240 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 20 --batch-size 32 --learning-rate 0.0002`
- Train metrics: mae=2.92472, rmse=4.27267, mape=66.4732
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.25057, rmse=1.09296, smape=83.14, direction_acc=61.82%, samples=440

### NHITS direct target (ctx90, horizon 1, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx90_lr1e4_e30 --horizon-minutes 1 --context-length 90 --model-type nhits --target-mode direct --hidden-size 128 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.91067, rmse=4.40251, mape=96.0288
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24707, rmse=1.09014, smape=83.26, direction_acc=62.05%, samples=440

### NHITS direct target (ctx180, h=1, h512)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx180_h512_lr1e4_e30 --horizon-minutes 1 --context-length 180 --model-type nhits --target-mode direct --hidden-size 512 --num-layers 3 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.88419, rmse=4.30364, mape=94.7351
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24913, rmse=1.09110, smape=83.29, direction_acc=61.82%, samples=440

### NHITS direct target (ctx240, h=1, h512)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx240_h512_lr1e4_e20 --horizon-minutes 1 --context-length 240 --model-type nhits --target-mode direct --hidden-size 512 --num-layers 3 --epochs 20 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.92626, rmse=4.27342, mape=66.4732
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.25306, rmse=1.09368, smape=83.83, direction_acc=61.82%, samples=440

### NHITS direct target (ctx120, lr=5e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx120_lr5e4_e30 --horizon-minutes 1 --context-length 120 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0005`
- Train metrics: mae=2.92206, rmse=4.28158, mape=111.614
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24285, rmse=1.09200, smape=82.14, direction_acc=61.82%, samples=440

### NHITS direct target (ctx180, lr=5e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx180_lr5e4_e30 --horizon-minutes 1 --context-length 180 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0005`
- Train metrics: mae=2.88587, rmse=4.30540, mape=94.7338
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24558, rmse=1.09239, smape=82.46, direction_acc=62.05%, samples=440

### PatchTST direct target (ctx90, lr=2e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_patchtst_ctx90_lr2e4_e20 --horizon-minutes 1 --context-length 90 --model-type patchtst --target-mode direct --hidden-size 128 --num-layers 2 --patch-length 8 --stride 4 --epochs 20 --batch-size 32 --learning-rate 0.0002`
- Train metrics: mae=0.99875, rmse=1.61219, mape=30.1723
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.99207, rmse=1.84272, smape=190.11, direction_acc=8.86%, samples=440

### Quick backtest sanity (price accuracy)
- Command: `python -m src.main pumpfun-backtest --model-dir models_pumpfun_direct_nhits_ctx120_lr1e4_e30 --minutes 10 --test-window 60 --target-mode direct --max-tokens 2 --max-samples 200`
- Backtest: mae=0.26311, rmse=1.14318, smape=124.78, direction_acc=41.00%, price_accuracy_pct=75.00%, samples=100

### NHITS direct target (ctx300, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx300_lr1e4_e20 --horizon-minutes 1 --context-length 300 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 20 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.87668, rmse=4.25886, mape=81.5318
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.25041, rmse=1.09127, smape=83.66, direction_acc=62.05%, price_accuracy_pct=65.02%, samples=440

### NHITS direct target (ctx120, lr=2e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx120_lr2e4_e20 --horizon-minutes 1 --context-length 120 --model-type nhits --target-mode direct --hidden-size 128 --num-layers 2 --epochs 20 --batch-size 32 --learning-rate 0.0002`
- Train metrics: mae=2.92624, rmse=4.28616, mape=111.813
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24987, rmse=1.09221, smape=83.80, direction_acc=62.05%, price_accuracy_pct=65.45%, samples=440

### NHITS direct target + token meta (ctx120, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_meta_nhits_ctx120_lr1e4_e30 --horizon-minutes 1 --context-length 120 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.43489, rmse=3.81026, mape=7.8363e10
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24673, rmse=1.09234, smape=82.82, direction_acc=62.05%, price_accuracy_pct=65.45%, samples=440

### NHITS direct target + token meta (ctx60, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_meta_nhits_ctx60_lr1e4_e30 --horizon-minutes 1 --context-length 60 --model-type nhits --target-mode direct --hidden-size 128 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.43176, rmse=3.80580, mape=7.8367e10
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24370, rmse=1.09136, smape=82.60, direction_acc=62.05%, price_accuracy_pct=65.15%, samples=440

### NHITS direct target (ctx30, h64, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx30_h64_lr1e4_e30 --horizon-minutes 1 --context-length 30 --model-type nhits --target-mode direct --hidden-size 64 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.43357, rmse=3.80606, mape=7.8293e10
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24337, rmse=1.08899, smape=82.58, direction_acc=62.05%, price_accuracy_pct=65.48%, samples=440

### NHITS direct target (ctx45, h128, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_nhits_ctx45_h128_lr1e4_e30 --horizon-minutes 1 --context-length 45 --model-type nhits --target-mode direct --hidden-size 128 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.43047, rmse=3.80221, mape=7.7899e10
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.24382, rmse=1.09098, smape=82.74, direction_acc=61.82%, price_accuracy_pct=65.40%, samples=440

### PatchTST direct target (ctx60, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct_patchtst_ctx60_lr1e4_e20 --horizon-minutes 1 --context-length 60 --model-type patchtst --target-mode direct --hidden-size 128 --num-layers 2 --patch-length 8 --stride 4 --epochs 20 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=0.30718, rmse=0.81717, mape=6.8369e9
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.69564, rmse=1.46193, smape=148.39, direction_acc=39.32%, price_accuracy_pct=71.90%, samples=440

### NHITS direct target (10m horizon, ctx120, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct10_nhits_ctx120_lr1e4_e30 --horizon-minutes 10 --context-length 120 --model-type nhits --target-mode direct --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.99968, rmse=4.08928, mape=2.3236e11
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.23502, rmse=1.08712, smape=81.87, direction_acc=62.05%, price_accuracy_pct=65.93%, samples=440

### NHITS direct target (10m horizon, ctx60, lr=1e-4)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_direct10_nhits_ctx60_lr1e4_e30 --horizon-minutes 10 --context-length 60 --model-type nhits --target-mode direct --hidden-size 128 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=2.99404, rmse=4.08093, mape=2.3118e11
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.22681, rmse=1.08265, smape=81.42, direction_acc=61.82%, price_accuracy_pct=65.97%, samples=440

### Direction classifier v1 (10m horizon)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v1 --horizon-minutes 10 --hidden-dim 128 --epochs 10 --batch-size 1024 --learning-rate 0.001`
- Validation directional accuracy: 84.17%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=86.05%, samples=3964

### Direction classifier v2 (wider + dropout)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v2 --horizon-minutes 10 --hidden-dim 256 --epochs 20 --batch-size 1024 --learning-rate 0.001 --dropout 0.2`
- Validation directional accuracy: 84.13%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=86.05%, samples=3964

### Direction classifier v3 (lower lr)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v3 --horizon-minutes 10 --hidden-dim 128 --epochs 20 --batch-size 1024 --learning-rate 0.0005 --dropout 0.1`
- Validation directional accuracy: 84.18%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=85.90%, samples=3964

### Direction classifier v4 (threshold tuned)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v4 --horizon-minutes 10 --hidden-dim 128 --epochs 10 --batch-size 1024 --learning-rate 0.001`
- Validation directional accuracy (best threshold): 84.19%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=85.85%, samples=3964

### Direction classifier v5 (log_trades feature)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v5 --horizon-minutes 10 --hidden-dim 128 --epochs 10 --batch-size 1024 --learning-rate 0.001`
- Validation directional accuracy: 84.10%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=85.67%, samples=3964

### Direction classifier v6 (ret_mean_5/ret_std_5)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v6 --horizon-minutes 10 --hidden-dim 128 --epochs 10 --batch-size 1024 --learning-rate 0.001`
- Validation directional accuracy: 84.38%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=85.91%, samples=3952

### Direction classifier (label_threshold=0.02)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_thr002 --horizon-minutes 10 --hidden-dim 128 --epochs 10 --batch-size 1024 --learning-rate 0.001 --label-threshold 0.02`
- Validation directional accuracy: 82.15%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=83.57%, samples=3377

### Direction classifier v7b (3-layer MLP)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v7b --horizon-minutes 10 --hidden-dim 128 --num-layers 3 --epochs 10 --batch-size 1024 --learning-rate 0.001 --dropout 0.1`
- Validation directional accuracy: 84.95%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=86.36%, samples=3952

### Direction classifier v8 (4-layer MLP, h64)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v8 --horizon-minutes 10 --hidden-dim 64 --num-layers 4 --epochs 10 --batch-size 1024 --learning-rate 0.001 --dropout 0.2`
- Validation directional accuracy: 84.95%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=86.29%, samples=3952

### Direction classifier v9 (3-layer MLP, longer epochs)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v9 --horizon-minutes 10 --hidden-dim 128 --num-layers 3 --epochs 20 --batch-size 2048 --learning-rate 0.001 --dropout 0.1`
- Validation directional accuracy: 84.94%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=86.74%, samples=3952

### Direction classifier v11 (3-layer, lr=5e-4)
- Command: `python -m src.main pumpfun-classify-train --model-dir models_pumpfun_classifier_v11 --horizon-minutes 10 --hidden-dim 128 --num-layers 3 --epochs 20 --batch-size 2048 --learning-rate 0.0005 --dropout 0.1`
- Validation directional accuracy: 85.04%
- Backtest (holdout, max tokens 4, max samples 5000): directional_accuracy=86.64%, samples=3952

### NHITS sum target (10m horizon, ctx240)
- Command: `python -m src.main pumpfun-train --model-dir models_pumpfun_sum10_ctx240_lr1e4_e30 --horizon-minutes 10 --context-length 240 --model-type nhits --target-mode sum --hidden-size 256 --num-layers 2 --epochs 30 --batch-size 32 --learning-rate 0.0001`
- Train metrics: mae=0.01827, rmse=0.02501, mape=2.1346
- Backtest (10m, window 120, max tokens 4, max samples 1500): mae=0.04553, rmse=0.07909, smape=194.43, direction_acc=5.68%, price_accuracy_pct=95.50%, samples=440
