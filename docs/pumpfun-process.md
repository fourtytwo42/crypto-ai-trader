## Pump.fun prediction process (ad-hoc by mint)

This is the exact workflow used to generate 5m/10m forecasts from the pump.fun trade stream in this branch.

### Data flow

1) Resolve mint address to `tokens.id`.
2) Pull all trades for that token.
3) Normalize trades:
   - fill missing `price_usd` from `amount_usd / amount_sol` when possible
   - fill missing `amount_usd` from `price_usd * amount_sol` when possible
   - fallback to SOL/USD lookup when needed
4) Aggregate into 1-minute candles.
5) Build minute features + token metadata features.
6) Run the pump.fun regression model and sum horizon returns for 5m/10m.

### Example (one-off in CLI)

```bash
./venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np

from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.models import PumpToken
from pumpfun_train.pipeline import (
    _load_trades_for_token,
    _normalize_trades,
    _price_lookup_with_cache,
    build_minute_candles,
)
from pumpfun_train.recent_features import build_recent_feature_df
from pumpfun_train.data import PUMPFUN_FEATURE_COLUMNS
from pumpfun_train.model_loader import load_model_artifacts
from pumpfun_train.forecast import forecast_next_horizon

mint = "YOUR_MINT_ADDRESS"
model_dir = Path("pumpfun_train/models/regression/h10")

bundle = load_model_artifacts(model_dir)
config_meta = bundle.metadata.get("config", {})
model_horizon = int(config_meta.get("horizon", 10))
context_length = int(config_meta.get("context_length", 336))

db = get_pumpfun_db_manager()
with db.session() as session:
    token = session.query(PumpToken).filter(PumpToken.mint_address == mint).one_or_none()
    if token is None:
        raise SystemExit("TOKEN_NOT_FOUND")
    token_id = token.id
    token_meta = {
        "created_timestamp": token.created_timestamp,
        "king_of_the_hill_timestamp": token.king_of_the_hill_timestamp,
        "completed": token.completed,
    }
    trades_df = _load_trades_for_token(session, token_id)
    if trades_df.empty:
        raise SystemExit("NO_TRADES")
    price_lookup = _price_lookup_with_cache(session)
    normalized = _normalize_trades(trades_df, price_lookup)

candles = build_minute_candles(normalized)
features_df, _ = build_recent_feature_df(candles, token_meta=token_meta)
history_df = features_df.tail(context_length).reset_index(drop=True)
feature_cols = [
    col
    for col in [*PUMPFUN_FEATURE_COLUMNS, "log_close", "log_volume"]
    if col in history_df.columns
]

preds = forecast_next_horizon(
    bundle.model,
    history_df,
    target_col="return",
    feature_cols=feature_cols,
    horizon=model_horizon,
)

preds = np.asarray(preds, dtype=float)
current_price = float(candles["close"].iloc[-1])

for minutes in (5, 10):
    pred_return = float(np.sum(preds[:minutes]))
    predicted_price = float(current_price * np.exp(pred_return))
    change = predicted_price - current_price
    pct = (change / current_price) * 100 if current_price else 0.0
    print(f"{minutes}m: current={current_price:.6f} predicted={predicted_price:.6f} change={change:.6f} pct={pct:.2f}%")
PY
```

### API surface (pumpfun_api)

The self-contained FastAPI service exposes:

- `GET /predict?mint=...&minutes=1..60` for minute projections (minutes > 10 are extrapolated from the last predicted returns).
- `GET /token?mint=...` for token metadata + latest price/market cap.
- `GET /candles?mint=...&limit=...` for minute candles used by the UI.

### Notes

- This uses the pump.fun regression model (NHITS). Directional classification lives in a separate classifier pipeline.
- Candles/features are derived from trades on demand, so this path always uses the latest trade stream.
- The pump.fun frontend uses price or market cap views; market cap assumes a 1B token supply.
