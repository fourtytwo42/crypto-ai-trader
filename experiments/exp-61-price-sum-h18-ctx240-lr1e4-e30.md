# Experiment 61 - Price model (sum target) horizon 18m

## Goal
Evaluate sum-target regression price accuracy at 18 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum18_ctx240_lr1e4_e30`
- Horizon: 18 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.017508592856383516, 'rmse': 0.02414971855577009, 'mape': 13.91406485142934}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.11476832766531642, 'rmse': 0.29496649494226024, 'smape': 180.2089044364483, 'direction_accuracy': 24.509803921568626, 'price_accuracy_pct': 90.06004145404492, 'samples': 1020}

## Results
- Training metrics: {'mae': 0.017508592856383516, 'rmse': 0.02414971855577009, 'mape': 13.91406485142934}
- Backtest metrics: {'mae': 0.11476832766531642, 'rmse': 0.29496649494226024, 'smape': 180.2089044364483, 'direction_accuracy': 24.509803921568626, 'price_accuracy_pct': 90.06004145404492, 'samples': 1020}
