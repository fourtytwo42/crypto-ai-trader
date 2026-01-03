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
