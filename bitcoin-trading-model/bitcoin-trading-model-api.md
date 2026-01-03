# Bitcoin Trading Model - API Specifications

**Complete API endpoint specifications, request/response formats, authentication, and error handling.**

## Table of Contents

1. [API Overview](#api-overview)
2. [Base URL](#base-url)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
5. [Request/Response Models](#requestresponse-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

## API Overview

The Bitcoin Trading Model API provides programmatic access to:
- Model predictions
- Model metadata
- Health checks
- Backtest results

**Framework:** FastAPI
**Documentation:** Automatic OpenAPI/Swagger docs at `/docs`
**Version:** 1.0.0

## Base URL

**Development:**
```
http://localhost:8000
```

**Production:**
```
https://api.yourdomain.com
```

## Authentication

**Current:** None (can add API keys later)

**Future:** API key authentication via header:
```
X-API-Key: your-api-key
```

## Endpoints

### 1. Health Check

**GET** `/health`

Check if API is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-28T10:30:00Z",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: API is healthy

### 2. List Models

**GET** `/models`

Get list of all available models.

**Query Parameters:**
- `active` (optional, boolean): Only return active models

**Response:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "model_v1",
      "model_type": "patchtst",
      "version": "1.0.0",
      "created_at": "2025-01-28T10:00:00Z",
      "metrics": {
        "val_mae": 0.012,
        "val_mape": 1.5
      }
    }
  ],
  "total": 1
}
```

**Status Codes:**
- `200 OK`: Success

### 3. Get Model Details

**GET** `/models/{model_id}`

Get detailed information about a specific model.

**Path Parameters:**
- `model_id` (integer): Model ID

**Response:**
```json
{
  "id": 1,
  "name": "model_v1",
  "model_type": "patchtst",
  "version": "1.0.0",
  "config": {
    "context_length": 128,
    "horizon": 1,
    "hidden_size": 512,
    "num_layers": 6,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 100,
    "quantiles": [0.1, 0.5, 0.9]
  },
  "train_start": "2013-10-06T00:00:00Z",
  "train_end": "2019-12-31T23:59:59Z",
  "val_start": "2020-01-01T00:00:00Z",
  "val_end": "2021-12-31T23:59:59Z",
  "test_start": "2022-01-01T00:00:00Z",
  "test_end": "2023-12-31T23:59:59Z",
  "metrics": {
    "val_mae": 0.012,
    "val_mape": 1.5,
    "val_sharpe": 1.2,
    "test_mae": 0.015,
    "test_mape": 2.0,
    "test_sharpe": 0.8
  },
  "file_path": "models/model_v1.pt",
  "scaler_path": "models/scaler_v1.pkl",
  "created_at": "2025-01-28T10:00:00Z",
  "updated_at": "2025-01-28T10:00:00Z"
}
```

**Status Codes:**
- `200 OK`: Success
- `404 Not Found`: Model not found

### 4. Generate Prediction

**POST** `/predict`

Generate a prediction using the latest data.

**Request Body:**
```json
{
  "model_id": 1,
  "threshold": 0.005,
  "use_latest": true
}
```

**Request Fields:**
- `model_id` (integer, required): Model ID to use
- `threshold` (float, optional, default: 0.005): Signal threshold (0.5% default)
- `use_latest` (boolean, optional, default: true): Use latest data from database

**Alternative Request (with custom data):**
```json
{
  "model_id": 1,
  "threshold": 0.005,
  "use_latest": false,
  "data": {
    "features": [
      {
        "return": 0.01,
        "range": 0.02,
        "body": 0.005,
        "dlog_volume": 0.1,
        "ret_mean_7": 0.005,
        "ret_std_7": 0.015,
        "ret_mean_30": 0.003,
        "ret_std_30": 0.02
      }
    ]
  }
}
```

**Response:**
```json
{
  "model_id": 1,
  "model_name": "model_v1",
  "prediction": {
    "q10": -0.02,
    "q50": 0.01,
    "q90": 0.04
  },
  "signal": "long",
  "threshold": 0.005,
  "prediction_timestamp": "2025-01-29T00:00:00Z",
  "timestamp": "2025-01-28T10:30:00Z",
  "confidence": 0.65
}
```

**Response Fields:**
- `model_id`: Model used for prediction
- `model_name`: Model name
- `prediction`: Quantile predictions (q10, q50, q90)
- `signal`: Generated signal ("long", "short", "flat")
- `threshold`: Threshold used
- `prediction_timestamp`: What timestamp we're predicting (next day)
- `timestamp`: When prediction was made
- `confidence`: Confidence score (0-1)

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request (missing model_id, invalid threshold)
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Prediction failed

### 5. Get Latest Predictions

**GET** `/predictions/latest`

Get the most recent predictions for a model.

**Query Parameters:**
- `model_id` (integer, required): Model ID
- `limit` (integer, optional, default: 10): Number of predictions to return

**Response:**
```json
{
  "predictions": [
    {
      "id": 100,
      "prediction_timestamp": "2025-01-29T00:00:00Z",
      "q10": -0.02,
      "q50": 0.01,
      "q90": 0.04,
      "signal": "long",
      "threshold": 0.005,
      "timestamp": "2025-01-28T10:30:00Z"
    }
  ],
  "total": 10
}
```

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Missing model_id
- `404 Not Found`: Model not found

### 6. List Backtests

**GET** `/backtests`

Get list of all backtests.

**Query Parameters:**
- `model_id` (integer, optional): Filter by model ID
- `limit` (integer, optional, default: 20): Number of results

**Response:**
```json
{
  "backtests": [
    {
      "id": 1,
      "model_id": 1,
      "model_name": "model_v1",
      "name": "backtest_2023",
      "start_date": "2022-01-01T00:00:00Z",
      "end_date": "2023-12-31T23:59:59Z",
      "metrics": {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "win_rate": 0.58,
        "avg_win": 0.02,
        "avg_loss": -0.015,
        "max_drawdown": -0.08,
        "num_trades": 150,
        "turnover": 0.5
      },
      "created_at": "2025-01-28T11:00:00Z"
    }
  ],
  "total": 1
}
```

**Status Codes:**
- `200 OK`: Success

### 7. Get Backtest Details

**GET** `/backtests/{backtest_id}`

Get detailed backtest results including trades.

**Path Parameters:**
- `backtest_id` (integer): Backtest ID

**Query Parameters:**
- `include_trades` (boolean, optional, default: false): Include individual trades

**Response:**
```json
{
  "id": 1,
  "model_id": 1,
  "model_name": "model_v1",
  "name": "backtest_2023",
  "start_date": "2022-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "train_window": 1825,
  "test_window": 365,
  "fees": 0.001,
  "slippage": 0.0005,
  "metrics": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "win_rate": 0.58,
    "avg_win": 0.02,
    "avg_loss": -0.015,
    "max_drawdown": -0.08,
    "num_trades": 150,
    "turnover": 0.5
  },
  "trades": [
    {
      "id": 1,
      "entry_timestamp": "2022-01-05T00:00:00Z",
      "exit_timestamp": "2022-01-06T00:00:00Z",
      "signal": "long",
      "entry_price": 42000.0,
      "exit_price": 42500.0,
      "pnl": 500.0,
      "fees": 84.5,
      "slippage": 21.0,
      "net_pnl": 394.5
    }
  ],
  "created_at": "2025-01-28T11:00:00Z"
}
```

**Status Codes:**
- `200 OK`: Success
- `404 Not Found`: Backtest not found

## Request/Response Models

### Pydantic Models

```python
# src/api/models.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class PredictionRequest(BaseModel):
    model_id: int = Field(..., description="Model ID to use")
    threshold: float = Field(0.005, ge=0, le=0.1, description="Signal threshold")
    use_latest: bool = Field(True, description="Use latest data from database")
    data: Optional[dict] = Field(None, description="Custom feature data")

class QuantilePrediction(BaseModel):
    q10: float
    q50: float
    q90: float

class PredictionResponse(BaseModel):
    model_id: int
    model_name: str
    prediction: QuantilePrediction
    signal: str = Field(..., pattern="^(long|short|flat)$")
    threshold: float
    prediction_timestamp: datetime
    timestamp: datetime
    confidence: float = Field(..., ge=0, le=1)

class ModelInfo(BaseModel):
    id: int
    name: str
    model_type: str
    version: str
    created_at: datetime
    metrics: Optional[dict] = None

class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    num_trades: int
    turnover: float
```

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with ID 999 not found",
    "details": {}
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `MODEL_NOT_FOUND` | 404 | Model ID not found |
| `BACKTEST_NOT_FOUND` | 404 | Backtest ID not found |
| `INSUFFICIENT_DATA` | 400 | Not enough data for prediction |
| `PREDICTION_FAILED` | 500 | Model prediction error |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Example Error Responses

**400 Bad Request:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid threshold value. Must be between 0 and 0.1",
    "details": {
      "field": "threshold",
      "value": 0.5
    }
  }
}
```

**404 Not Found:**
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model with ID 999 not found",
    "details": {
      "model_id": 999
    }
  }
}
```

**500 Internal Server Error:**
```json
{
  "error": {
    "code": "PREDICTION_FAILED",
    "message": "Failed to generate prediction: Model file not found",
    "details": {}
  }
}
```

## Rate Limiting

**Current:** No rate limiting

**Future:** 
- 100 requests per minute per IP
- 1000 requests per hour per IP
- Rate limit headers in response:
  ```
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 95
  X-RateLimit-Reset: 1640995200
  ```

## Examples

### Python Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Generate prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "model_id": 1,
        "threshold": 0.005,
        "use_latest": True
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Signal: {data['signal']}")
    print(f"Prediction: {data['prediction']}")
else:
    print(f"Error: {response.json()}")
```

### cURL Example

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Generate prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": 1,
    "threshold": 0.005,
    "use_latest": true
  }'

# Get backtest results
curl http://localhost:8000/backtests/1?include_trades=true
```

### JavaScript Example

```javascript
// Generate prediction
async function getPrediction(modelId, threshold = 0.005) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model_id: modelId,
      threshold: threshold,
      use_latest: true
    })
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return await response.json();
}

// Usage
getPrediction(1, 0.005)
  .then(data => {
    console.log('Signal:', data.signal);
    console.log('Prediction:', data.prediction);
  })
  .catch(error => console.error('Error:', error));
```

