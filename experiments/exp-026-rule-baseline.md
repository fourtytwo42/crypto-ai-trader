# Experiment 026 - Rule baseline (ret_mean_5 sign)

## Goal
Evaluate a simple heuristic as a baseline direction predictor.

## Setup
- Holdout tokens: from `models_pumpfun_classifier_v9/holdout_tokens.txt` (all)
- Direction rule: sign of `ret_mean_5` vs 10-minute return sign

## Command
```bash
./venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np

from src.pumpfun.data import prepare_pumpfun_training_data, add_return_target
from src.pumpfun.db import get_pumpfun_db_manager

model_dir = Path('models_pumpfun_classifier_v9')
holdout_tokens = [line.strip() for line in (model_dir / 'holdout_tokens.txt').read_text().splitlines() if line.strip()]

with get_pumpfun_db_manager().session() as session:
    training = prepare_pumpfun_training_data(session, holdout_tokens, normalize=True)

df = add_return_target(training.df, 10).dropna().reset_index(drop=True)

rule = np.sign(df['ret_mean_5'].to_numpy())
actual = np.sign(df['return_horizon'].to_numpy())
acc = (rule == actual).mean() * 100

print(f"Samples: {len(df)}")
print(f"Rule accuracy (ret_mean_5 sign): {acc:.2f}%")
PY
```

## Results
- Samples: 11765
- Direction accuracy: `68.55%`

## Notes
- Simple momentum rule is much worse than learned classifier.
