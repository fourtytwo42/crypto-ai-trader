# Experiment 50 - Price model (sum target) horizon 7m

## Goal
Evaluate sum-target regression price accuracy at 7 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum07_ctx240_lr1e4_e30`
- Horizon: 7 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.0189011589950661, 'rmse': 0.025517213747666335, 'mape': 3.194271792359312}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.06717673830417871, 'rmse': 0.12559512035334036, 'smape': 187.26590905000262, 'direction_accuracy': 18.4070796460177, 'price_accuracy_pct': 93.2257376380004, 'samples': 1130}

## Results
- Training metrics: {'mae': 0.0189011589950661, 'rmse': 0.025517213747666335, 'mape': 3.194271792359312}
- Backtest metrics: {'mae': 0.06717673830417871, 'rmse': 0.12559512035334036, 'smape': 187.26590905000262, 'direction_accuracy': 18.4070796460177, 'price_accuracy_pct': 93.2257376380004, 'samples': 1130}
