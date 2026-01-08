# Experiment 52 - Price model (sum target) horizon 9m

## Goal
Evaluate sum-target regression price accuracy at 9 minute horizon.

## Setup
- Model dir: `models_pumpfun_sum09_ctx240_lr1e4_e30`
- Horizon: 9 minutes
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
Training epochs [------------------------] 1/30Training epochs [#-----------------------] 2/30Pump.fun training complete. Metrics: {'mae': 0.017735248907776672, 'rmse': 0.02372168963945839, 'mape': 2.494329502806634}
Holdout tokens saved: 12
Backtest metrics: {'mae': 0.06971483826539984, 'rmse': 0.1490210264087355, 'smape': 186.5237702552224, 'direction_accuracy': 18.10810810810811, 'price_accuracy_pct': 93.35225840440589, 'samples': 1110}

## Results
- Training metrics: {'mae': 0.017735248907776672, 'rmse': 0.02372168963945839, 'mape': 2.494329502806634}
- Backtest metrics: {'mae': 0.06971483826539984, 'rmse': 0.1490210264087355, 'smape': 186.5237702552224, 'direction_accuracy': 18.10810810810811, 'price_accuracy_pct': 93.35225840440589, 'samples': 1110}
