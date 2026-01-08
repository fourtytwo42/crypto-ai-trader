# Experiment 47 - Price model (sum target) horizon 4m

## Goal
Evaluate sum-target regression price accuracy at 4 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum04_ctx240_lr1e4_e30`
- Horizon: 4 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.018770028250221976, 'rmse': 0.025198544313962037, 'mape': 3.3721827603321084}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.04323858164903648, 'rmse': 0.07655123828575508, 'smape': 190.02506946887007, 'direction_accuracy': 13.879310344827585, 'price_accuracy_pct': 95.52678382564125, 'samples': 1160}

## Results
- Training metrics: {'mae': 0.018770028250221976, 'rmse': 0.025198544313962037, 'mape': 3.3721827603321084}
- Backtest metrics: {'mae': 0.04323858164903648, 'rmse': 0.07655123828575508, 'smape': 190.02506946887007, 'direction_accuracy': 13.879310344827585, 'price_accuracy_pct': 95.52678382564125, 'samples': 1160}
