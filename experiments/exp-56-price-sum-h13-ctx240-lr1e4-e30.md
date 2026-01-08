# Experiment 56 - Price model (sum target) horizon 13m

## Goal
Evaluate sum-target regression price accuracy at 13 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum13_ctx240_lr1e4_e30`
- Horizon: 13 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.017745362165369715, 'rmse': 0.024125015539127773, 'mape': 2.963453658515353}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.09855450644599253, 'rmse': 0.21480934629020476, 'smape': 184.91711365353044, 'direction_accuracy': 21.02803738317757, 'price_accuracy_pct': 90.52166776043978, 'samples': 1070}

## Results
- Training metrics: {'mae': 0.017745362165369715, 'rmse': 0.024125015539127773, 'mape': 2.963453658515353}
- Backtest metrics: {'mae': 0.09855450644599253, 'rmse': 0.21480934629020476, 'smape': 184.91711365353044, 'direction_accuracy': 21.02803738317757, 'price_accuracy_pct': 90.52166776043978, 'samples': 1070}
