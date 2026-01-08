# Pump.fun Prediction API (self-contained)

A minimal FastAPI service that loads bundled pump.fun regression models and returns minute forecasts for a mint.

## Contents

- `app.py` - FastAPI app
- `predictor.py` - end-to-end trade → candle → feature → prediction
- `models/` - bundled model artifacts (regression + classifier)

## Environment

Set `PUMPFUN_DATABASE_URL` to the pump.fun Postgres connection string.
You can place it in `pumpfun_api/.env` (this repo copies the root `.env` as a starting point).

## Run

```bash
# from pumpfun_api/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# set PUMPFUN_DATABASE_URL in pumpfun_api/.env
python -m uvicorn app:app --host 0.0.0.0 --port 8081
```

## API

- `GET /predict?mint=...&minutes=1..60`
- `GET /token?mint=...`
- `GET /candles?mint=...&limit=...`

Returns per-minute predictions from 1..minutes. Requests beyond the trained horizon are extrapolated from the last predicted returns.
