# Experiment 55 - Price model (sum target) horizon 12m

## Goal
Evaluate sum-target regression price accuracy at 12 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum12_ctx240_lr1e4_e30`
- Horizon: 12 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.018395818236235078, 'rmse': 0.02437708958719107, 'mape': 1.6343073649061615}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.08500489761673677, 'rmse': 0.1967958750000055, 'smape': 186.31678078189893, 'direction_accuracy': 20.74074074074074, 'price_accuracy_pct': 91.9441065024642, 'samples': 1080}

## Results
- Training metrics: {'mae': 0.018395818236235078, 'rmse': 0.02437708958719107, 'mape': 1.6343073649061615}
- Backtest metrics: {'mae': 0.08500489761673677, 'rmse': 0.1967958750000055, 'smape': 186.31678078189893, 'direction_accuracy': 20.74074074074074, 'price_accuracy_pct': 91.9441065024642, 'samples': 1080}
