## KuCoin Hourly Backfill

We backfill candles using the public KuCoin REST endpoint:
`GET https://api.kucoin.com/api/v1/market/candles`

Response format (newest -> oldest):
```
[ startTime, open, close, high, low, volume, turnover ]
```

Implementation:
- `src/data/kucoin_client.py` handles paging (1500 candles per request).
- `src/data/pipeline.py` loads KuCoin candles into Postgres.

Example (6 months of hourly BTC-USDT into Postgres):
```
source venv/bin/activate && python - <<'PY'
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data.pipeline import DataPipeline

engine = create_engine("postgresql://trading_user:password@localhost:5432/bitcoin_trading")
Session = sessionmaker(bind=engine)

now = int(time.time())
start_at = now - int(6 * 30 * 86400)

with Session() as session:
    pipeline = DataPipeline(session)
    count = pipeline.load_kucoin_to_database(
        symbol="BTC-USDT",
        timeframe="1hour",
        start_at=start_at,
        end_at=now,
        replace_existing=True,
    )
    print(f"loaded {count} candles")
PY
```
