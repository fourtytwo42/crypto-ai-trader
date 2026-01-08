# Experiment 54 - Price model (sum target) horizon 11m

## Goal
Evaluate sum-target regression price accuracy at 11 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum11_ctx240_lr1e4_e30`
- Horizon: 11 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.01804078193784498, 'rmse': 0.02422880741497109, 'mape': 2.962014308200305}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.0804214714580055, 'rmse': 0.1775025114603444, 'smape': 184.5088580715376, 'direction_accuracy': 19.357798165137616, 'price_accuracy_pct': 92.54303837669934, 'samples': 1090}

## Results
- Training metrics: {'mae': 0.01804078193784498, 'rmse': 0.02422880741497109, 'mape': 2.962014308200305}
- Backtest metrics: {'mae': 0.0804214714580055, 'rmse': 0.1775025114603444, 'smape': 184.5088580715376, 'direction_accuracy': 19.357798165137616, 'price_accuracy_pct': 92.54303837669934, 'samples': 1090}
