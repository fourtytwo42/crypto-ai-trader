# Crypto AI Trader

An AI-powered cryptocurrency price prediction system using NHITS time-series forecasting models trained on hourly candle data from KuCoin.

## Overview

Crypto AI Trader uses deep learning (NHITS neural network) to predict cryptocurrency price direction and magnitude. The system achieves **~95% price accuracy** and **~98% directional accuracy** on holdout tests across BTC, ETH, LTC, and XRP.

## Current Best Model Performance

| Symbol | Directional Accuracy | Price Accuracy |
|--------|---------------------|----------------|
| BTC-USDT | 100.00% | 96.69% |
| ETH-USDT | 100.00% | 94.48% |
| LTC-USDT | 95.83% | 94.38% |
| XRP-USDT | 95.83% | 94.56% |
| **Average** | **97.92%** | **95.03%** |

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- CUDA-capable GPU (optional, for faster training)
- `structlog` (install via `pip install structlog`; required by the CLI)

### Installation

```bash
# Clone and enter directory
git clone <repo-url>
cd crypto-ai-trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials
```

### Database Setup

```bash
# Create PostgreSQL database
createdb bitcoin_trading

# Apply migrations
alembic upgrade head
```

### Load Historical Data

```bash
# Load hourly data from KuCoin (8 years of history)
python -m src.main load-hourly --symbols BTC-USDT,ETH-USDT,LTC-USDT,XRP-USDT --years-back 8
```

## Pump.fun mint-level predictions

There is a pump.fun-only flow that pulls trades, normalizes them, builds 1-minute candles/features, and runs the minute model for 5m/10m forecasts. This is isolated from the BTC/ETH/LTC/SOL pipeline.

See `docs/pumpfun-process.md` for the step-by-step process and the one-off CLI snippet.

## Usage

### Quick Prediction (Recommended)

`quick-predict` now streams the most recent context window directly from KuCoin (no database writes) and runs inference for any symbol you specify. Under the hood it fetches just `context_length + 720` hours, normalizes the features on the fly, and uses the saved NHITS model to project `return_24h`.

```bash
# Inference only: pull fresh candles for the model's context and predict
python -m src.main quick-predict --hours 24

# Use quick-predict with a specific pair (no DB seed required)
python -m src.main quick-predict --hours 24 --symbols BTC-USDT

# Retrain the NHITS model and predict with the new weights
python -m src.main quick-predict --hours 24 --retrain

# Predict RTT for other KuCoin pairs
python -m src.main quick-predict --hours 24 --symbols ADA-USDT
```

```

**Sample Output:**
```
============================================================
PRICE PREDICTIONS (24h ahead)
============================================================
Model: Loaded from models_nhits_best
Generated: 2026-01-07T04:17:11+00:00
------------------------------------------------------------

BTC-USDT:
  Current Price:   $92,802.40
  Predicted Price: $93,025.08
  Change:          $+222.68 (+0.24%)
  Direction:       [UP]

ETH-USDT:
  Current Price:   $3,267.04
  Predicted Price: $3,277.85
  Change:          $+10.81 (+0.33%)
  Direction:       [UP]

XRP-USDT:
  Current Price:   $2.27
  Predicted Price: $2.24
  Change:          $-0.03 (-1.24%)
  Direction:       [DOWN]
============================================================
```

### Model Training

Train the best-performing NHITS model:

```bash
python -m src.main quick-predict --hours 24 --retrain
```

Or run the full holdout evaluation:

```bash
python -m src.main forecast-holdout-24h \
  --context-length 336 \
  --hidden-size 512 \
  --num-layers 3 \
  --patch-length 8 \
  --stride 4 \
  --loss-type huber \
  --epochs 50 \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --model-type nhits \
  --nhits-stack-types identity,identity,identity \
  --nhits-n-blocks 3,2,2 \
  --nhits-mlp-units "768|768;768|768;768|768" \
  --nhits-n-pool-kernel-size 2,2,1 \
  --nhits-n-freq-downsample 4,2,1 \
  --multi-asset
```

### Other Commands

```bash
# Load daily CSV data
python -m src.main load-data /path/to/data.csv --symbol BTC-USDT

# Load hourly data from KuCoin
python -m src.main load-hourly --symbols BTC-USDT,ETH-USDT --years-back 8

# Start REST API server
python -m src.main api --host 0.0.0.0 --port 8000

# Interactive menu
python -m src.main --menu

# View all commands
python -m src.main --help
```

## API

Start the API:

```bash
python -m src.main api --host 0.0.0.0 --port 8000
```

Example requests:

```bash
# Start a retrain job
curl -X POST http://localhost:8000/trainings \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "nhits",
    "context_length": 168,
    "horizon_hours": 24,
    "hidden_size": 512,
    "num_layers": 6,
    "epochs": 50
  }'

# Check training status
curl http://localhost:8000/trainings/1

# Run a cached forecast prediction
curl -X POST http://localhost:8000/forecast/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USDT",
    "hours": 24
  }'

# Fetch prediction history
curl http://localhost:8000/forecast/history/BTC-USDT?limit=50
```

## Architecture

### Source Layout

```
src/
├── main.py              # Entry point, structlog configuration
├── config.py            # Pydantic settings from environment
├── data/                # Data pipeline
│   ├── csv_loader.py    # Kraken CSV format loader
│   ├── feature_extractor.py  # Log returns, range, volume features
│   ├── kucoin_client.py # Exchange API client
│   └── pipeline.py      # Orchestrates data flow
├── database/            # PostgreSQL layer
│   ├── connection.py    # SQLAlchemy engine
│   ├── models.py        # ORM models
│   └── operations.py    # CRUD operations
├── training/            # Model training
│   ├── trainer.py       # Training orchestration
│   ├── model_factory.py # Creates NHITS/PatchTST models
│   └── config.py        # Training configuration
├── prediction/          # Inference
│   ├── model_loader.py  # Load saved models
│   └── predictor.py     # Generate forecasts
├── forecasting/         # Walk-forward evaluation
│   └── walk_forward_forecast.py
├── cli/                 # Terminal interface
│   ├── cli.py           # Click commands
│   └── commands.py      # Command implementations
└── api/                 # REST API
    └── main.py          # FastAPI app
```

### Best Model Configuration (NHITS)

| Parameter | Value |
|-----------|-------|
| Model Type | NHITS |
| Horizon | 1 (24h prediction) |
| Context Length | 336 hours (14 days) |
| Stack Types | identity, identity, identity |
| N Blocks | 3, 2, 2 |
| MLP Units | 768, 768 per stack |
| Pool Kernel Size | 2, 2, 1 |
| Freq Downsample | 4, 2, 1 |
| Learning Rate | 5e-5 |
| Epochs | 50 |
| Batch Size | 16 |

### Feature Engineering (8 Scale-Free Features)

1. `return` - log(Close_t / Close_{t-1})
2. `range` - log(High_t / Low_t)
3. `body` - log(Close_t / Open_t)
4. `dlog_volume` - log(Volume_t) - log(Volume_{t-1})
5. `ret_mean_7` - 7-day rolling mean of returns
6. `ret_std_7` - 7-day rolling std of returns
7. `ret_mean_30` - 30-day rolling mean of returns
8. `ret_std_30` - 30-day rolling std of returns

## Technology Stack

- **Python 3.11** - Core language
- **NeuralForecast 1.7** - NHITS/PatchTST models
- **PyTorch 2.1** - Deep learning backend
- **PostgreSQL 15** - Data storage
- **SQLAlchemy 2.0** - ORM
- **FastAPI** - REST API
- **Click** - CLI framework
- **structlog** - Structured logging

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | required |
| `TRAIN_DEVICE` | Training device (cuda/cpu) | cpu |
| `MODEL_DIR` | Model storage directory | models |

## Documentation

- [EXPERIMENTS.md](EXPERIMENTS.md) - Full experiment log and best model details
- [docs/postgres-setup.md](docs/postgres-setup.md) - Database setup guide
- [docs/kucoin-backfill.md](docs/kucoin-backfill.md) - Data backfill process
- [docs/training-workflow.md](docs/training-workflow.md) - Training procedures

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Linting
ruff check src tests

# Formatting
black src tests

# Type checking
mypy src
```

## License

MIT
