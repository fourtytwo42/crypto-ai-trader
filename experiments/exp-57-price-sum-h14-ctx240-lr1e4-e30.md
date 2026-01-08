# Experiment 57 - Price model (sum target) horizon 14m

## Goal
Evaluate sum-target regression price accuracy at 14 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum14_ctx240_lr1e4_e30`
- Horizon: 14 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.017798401065594053, 'rmse': 0.024378758085018014, 'mape': 2.862099308768445}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.09881692051561465, 'rmse': 0.2303958741755158, 'smape': 185.22794618871876, 'direction_accuracy': 22.641509433962266, 'price_accuracy_pct': 90.7500629670263, 'samples': 1060}

## Results
- Training metrics: {'mae': 0.017798401065594053, 'rmse': 0.024378758085018014, 'mape': 2.862099308768445}
- Backtest metrics: {'mae': 0.09881692051561465, 'rmse': 0.2303958741755158, 'smape': 185.22794618871876, 'direction_accuracy': 22.641509433962266, 'price_accuracy_pct': 90.7500629670263, 'samples': 1060}
