## Pump.fun Prediction API

The pump.fun API is served by `pumpfun_api/app.py`.

### Health

`GET /health`

### Token snapshot

`GET /token?mint=...`

Returns mint metadata plus the latest price and market cap estimate.

### Candles

`GET /candles?mint=...&limit=240`

Returns minute candles used by the UI (limit 10..2000).

### Predictions

`GET /predict?mint=...&minutes=10`

Returns per-minute predictions up to 60 minutes. Requests beyond the trained horizon are extrapolated from the last predicted returns.
