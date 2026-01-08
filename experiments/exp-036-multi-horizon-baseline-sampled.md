# Experiment 036 - Multi-horizon eval (sampled)

## Goal
Evaluate per-minute (1..20) direction + price accuracy for 20-minute model.

## Setup
- Model dir: `models_pumpfun_sum20_ctx240_lr1e4_e20`
- Horizons: 1..20
- Holdout tokens: first 4 tokens from holdout list
- Samples: 60 windows per token (240 total)
- Context length: 240
- Test window: 120

## Command
```bash
./venv/bin/python - <<'PY'
# multi-horizon eval (sampled)
PY
```

## Results
- 1m: direction=0.83% price=98.86% samples=240
- 2m: direction=1.25% price=98.08% samples=240
- 3m: direction=1.67% price=97.69% samples=240
- 4m: direction=1.67% price=97.29% samples=240
- 5m: direction=2.50% price=96.96% samples=240
- 6m: direction=2.92% price=96.67% samples=240
- 7m: direction=3.33% price=96.26% samples=240
- 8m: direction=2.92% price=95.83% samples=240
- 9m: direction=2.08% price=95.52% samples=240
- 10m: direction=2.50% price=95.33% samples=240
- 11m: direction=2.50% price=95.03% samples=240
- 12m: direction=2.92% price=94.76% samples=240
- 13m: direction=3.33% price=94.57% samples=240
- 14m: direction=3.75% price=94.28% samples=240
- 15m: direction=3.75% price=94.13% samples=240
- 16m: direction=3.75% price=93.93% samples=240
- 17m: direction=5.83% price=93.78% samples=240
- 18m: direction=8.33% price=93.66% samples=240
- 19m: direction=8.75% price=93.36% samples=240
- 20m: direction=6.67% price=92.97% samples=240

## Notes
- Price accuracy remains high, but direction accuracy from regression is unusable across horizons.
- Need separate per-horizon direction models or alternative direction metric.
