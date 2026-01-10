# Experiment 068 - pumpfun multi-horizon (patchtst sum20 ctx240)

## Goal
Train a single horizon-20 PatchTST model and evaluate price accuracy for minutes 1..20 on holdout tokens.

## Config
- model_type=patchtst
- horizon_minutes=20, context_length=240
- hidden_size=512, num_layers=4, patch_length=8, stride=4
- epochs=50, batch_size=32, lr=1e-4

## Result
- Training process killed by OS (likely OOM) shortly after start.
