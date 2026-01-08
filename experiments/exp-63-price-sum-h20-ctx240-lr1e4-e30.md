# Experiment 63 - Price model (sum target) horizon 20m

## Goal
Evaluate sum-target regression price accuracy at 20 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum20_ctx240_lr1e4_e30`
- Horizon: 20 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.017965307754499994, 'rmse': 0.025104810222992376, 'mape': 11.062775353154349}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.1248966589172907, 'rmse': 0.3240394492512541, 'smape': 179.99779399500147, 'direction_accuracy': 24.5, 'price_accuracy_pct': 89.29208132986203, 'samples': 1000}

## Results
- Training metrics: {'mae': 0.017965307754499994, 'rmse': 0.025104810222992376, 'mape': 11.062775353154349}
- Backtest metrics: {'mae': 0.1248966589172907, 'rmse': 0.3240394492512541, 'smape': 179.99779399500147, 'direction_accuracy': 24.5, 'price_accuracy_pct': 89.29208132986203, 'samples': 1000}
