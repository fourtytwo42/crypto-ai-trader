# Experiment 48 - Price model (sum target) horizon 5m

## Goal
Evaluate sum-target regression price accuracy at 5 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum05_ctx240_lr1e4_e30`
- Horizon: 5 minutes
- Context length: 240
- Model: NHITS
- Target mode: sum
- Hidden size: 256
- Layers: 2
- Epochs: 30
- Batch size: 32
- Learning rate: 1e-4
- Holdout count: 12
- Backtest: test_window=120, max_tokens=12, max_samples=5000

## Commands
┏━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name         ┃ Type          ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ padder_train │ ConstantPad1d │      0 │ train │     0 │
│ 1 │ loss         │ HuberLoss     │      0 │ train │     0 │
│ 2 │ scaler       │ TemporalNorm  │      0 │ train │     0 │
│ 3 │ blocks       │ ModuleList    │  7.2 M │ train │     0 │
└───┴──────────────┴───────────────┴────────┴───────┴───────┘
Trainable params: 7.2 M                                                         
Non-trainable params: 0                                                         
Total params: 7.2 M                                                             
Total estimated model params size (MB): 28                                      
Modules in train mode: 34                                                       
Modules in eval mode: 0                                                         
Total FLOPs: 0                                                                  
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.019184186998627565, 'rmse': 0.02545808371796806, 'mape': 3.8310212271374344}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.059792237512254966, 'rmse': 0.09902182631908421, 'smape': 189.02195555746232, 'direction_accuracy': 16.434782608695652, 'price_accuracy_pct': 93.77948819490663, 'samples': 1150}

## Results
- Training metrics: {'mae': 0.019184186998627565, 'rmse': 0.02545808371796806, 'mape': 3.8310212271374344}
- Backtest metrics: {'mae': 0.059792237512254966, 'rmse': 0.09902182631908421, 'smape': 189.02195555746232, 'direction_accuracy': 16.434782608695652, 'price_accuracy_pct': 93.77948819490663, 'samples': 1150}
