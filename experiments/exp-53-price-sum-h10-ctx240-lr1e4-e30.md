# Experiment 53 - Price model (sum target) horizon 10m

## Goal
Evaluate sum-target regression price accuracy at 10 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum10_ctx240_lr1e4_e30`
- Horizon: 10 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.018058799551629798, 'rmse': 0.02482643405088071, 'mape': 1.7413200220180556}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.06970839140526881, 'rmse': 0.15878059383202237, 'smape': 184.53022034236696, 'direction_accuracy': 19.454545454545453, 'price_accuracy_pct': 93.41933819292537, 'samples': 1100}

## Results
- Training metrics: {'mae': 0.018058799551629798, 'rmse': 0.02482643405088071, 'mape': 1.7413200220180556}
- Backtest metrics: {'mae': 0.06970839140526881, 'rmse': 0.15878059383202237, 'smape': 184.53022034236696, 'direction_accuracy': 19.454545454545453, 'price_accuracy_pct': 93.41933819292537, 'samples': 1100}
