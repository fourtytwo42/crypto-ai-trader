# Experiments Log

Purpose: Track training/backtest runs, settings, and key metrics for comparing improvements.

## Legend
- Metrics: close_mape is lower better. close_within_0.5% / close_within_1% higher better.
- Windows: number of forecast windows evaluated.

## Multi-asset PatchTST (current best)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=640, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6252%, close_within_0.5%=0.6215, close_within_1%=0.8438
- Notes: new best across mape and within_0.5%/within_1%.

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

## Multi-asset PatchTST (return target + quadratic weighting, hidden 704, patch 6/stride 3)
- Config: train_size=35040, val_size=5256, test_size=4380, step_size=1095, epochs=50, context_length=168, hidden_size=704, num_layers=6, patch_length=6, stride=3, loss_type=huber, horizon_weight_mode=quadratic, close_target=return
- Windows: 8
- Metrics: close_mape=0.6737%, close_within_0.5%=0.6181, close_within_1%=0.8333
- Notes: wider hidden size did not beat hidden_size=640 best.

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

## Infra changes
- Added max_vram_gb cap (default 23.0) to training config and applied per-process CUDA memory fraction.

