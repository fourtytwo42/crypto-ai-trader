## API Endpoints

### Health

`GET /health`

### Models

- `GET /models` - list available models
- `GET /models/{model_id}` - fetch model details

### Training Jobs

- `POST /trainings` - start a retraining job
- `GET /trainings` - list recent training jobs
- `GET /trainings/{job_id}` - training job status

Example payload:

```json
{
  "model_type": "nhits",
  "context_length": 168,
  "horizon_hours": 24,
  "hidden_size": 512,
  "num_layers": 6,
  "patch_length": 16,
  "stride": 8,
  "learning_rate": 0.0001,
  "batch_size": 32,
  "epochs": 50,
  "loss_type": "mae",
  "symbols": ["BTC-USDT", "ETH-USDT"],
  "multi_asset": true,
  "name_prefix": "nhits_24h"
}
```

### Forecast Prediction

`POST /forecast/predict`

Example payload:

```json
{
  "symbol": "BTC-USDT",
  "hours": 24,
  "model_id": 1,
  "use_cache": true
}
```

The API caches predictions for up to 1 hour (per model, symbol, and data timestamp).

### Forecast History

`GET /forecast/history/{symbol}?model_id={id}&limit=100`

Returns the prediction history with predicted price, actual price (when available), accuracy, and model.
