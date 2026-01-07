## Training Workflow

Steps for hourly KuCoin data:
1) Backfill hourly candles into Postgres.
2) Extract and store features.
3) Prepare training data.
4) Train model and persist artifacts.

Extract and store features:
```
source venv/bin/activate && python - <<'PY'
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.pipeline import DataPipeline

engine = create_engine("postgresql://trading_user:password@localhost:5432/bitcoin_trading")
Session = sessionmaker(bind=engine)

with Session() as session:
    pipeline = DataPipeline(session)
    count = pipeline.extract_and_store_features()
    print(f"created {count} feature rows")
PY
```

Train a model:
```
source venv/bin/activate && python - <<'PY'
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.pipeline import prepare_training_data
from src.training.config import TrainingConfig
from src.training.trainer import train_model

engine = create_engine("postgresql://trading_user:password@localhost:5432/bitcoin_trading")
Session = sessionmaker(bind=engine)

with Session() as session:
    df, normalizer = prepare_training_data(session, normalize=True)
    config = TrainingConfig(model_type="nhits", horizon=48, context_length=128)
    result = train_model(
        config=config,
        train_df=df.iloc[:-200],
        val_df=df.iloc[-200:],
        model_dir="models",
        target_col="return",
        feature_cols=[c for c in df.columns if c not in {"timestamp", "unique_id", "return"}],
        scaler=normalizer,
    )
    print(result.metrics)
PY
```

Optimize model accuracy (12-candle horizon):
```
./venv/bin/python -m src.backtest.optimizer
```

Notes:
- Set `TrainingConfig(device="cuda")` to use GPU (RTX 3090) if available; it will
  fall back to CPU when CUDA is not present.
- The optimizer evaluates forecast accuracy (MAE/RMSE/MAPE) against `target_return`,
  which is `return` shifted by the forecast horizon.
- Progress logs show backtest windows, running average metrics, and per-config results.
- Training artifacts are only saved for explicit training runs; backtests do not
  overwrite existing models.

## Quick Prediction Workflow

`quick-predict` is now the production-facing entry point for live inference. Instead
of touching the historical database, the command streams the context window directly
from KuCoin for each symbol requested. Features are built and normalized on the fly
from the hourly candles, then the saved NHITS model predicts `return_24h`. No
database writes occur, so you can request any KuCoin pair—even ones you haven't
trained with.

```bash
source venv/bin/activate
python -m src.main quick-predict --hours 24 --symbols BTC-USDT
```

If you want to refresh the model, add `--retrain` to copy the latest training run
and reuse those weights for inference.
