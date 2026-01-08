# Experiment 035 - Multi-horizon eval (timeout)

## Goal
Evaluate per-minute (1..20) direction + price accuracy for the 20-minute horizon model.

## Setup
- Model dir: `models_pumpfun_sum20_ctx240_lr1e4_e20`
- Horizons: 1..20
- Holdout tokens: all in model holdout list
- Context length: 240
- Test window: 120
- Timeout: 300s

## Command
```bash
./venv/bin/python - <<'PY'
# multi-horizon eval script (see exp-036 for successful run)
PY
```

## Results
- Timed out at 300s before finishing.

## Notes
- Reduce tokens/samples for per-minute evaluation.
