# Experiment 60 - Price model (sum target) horizon 17m

## Goal
Evaluate sum-target regression price accuracy at 17 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum17_ctx240_lr1e4_e30`
- Horizon: 17 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.01773226067876481, 'rmse': 0.024400996223045018, 'mape': 18.593925462946597}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.10778746113513683, 'rmse': 0.27677632403353297, 'smape': 182.65161624130644, 'direction_accuracy': 23.20388349514563, 'price_accuracy_pct': 90.5451011970108, 'samples': 1030}

## Results
- Training metrics: {'mae': 0.01773226067876481, 'rmse': 0.024400996223045018, 'mape': 18.593925462946597}
- Backtest metrics: {'mae': 0.10778746113513683, 'rmse': 0.27677632403353297, 'smape': 182.65161624130644, 'direction_accuracy': 23.20388349514563, 'price_accuracy_pct': 90.5451011970108, 'samples': 1030}
