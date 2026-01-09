# Crypto AI Trader (Pump.fun)

Pump.fun mint-level forecasting system built around minute candles and NHITS models. This branch contains the pump.fun training pipeline, FastAPI prediction service, and Next.js UI.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (pump.fun trade + token tables)
- Node 18+ (for the frontend)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure environment

Set the pump.fun database connection string for training and API:

```bash
export PUMPFUN_DATABASE_URL="postgresql://user:password@localhost:5432/pumpfun_db"
```

### Train models

```bash
# Sync latest trades into Postgres
python -m pumpfun_train.cli_main pumpfun-sync

# Train all regression + classifier models
python pumpfun_train/train_all_models.py
```

Full workflow lives in `pumpfun_train/TRAINING_GUIDE.md`.

### Run API + Frontend

```bash
# API
cd pumpfun_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# set PUMPFUN_DATABASE_URL in pumpfun_api/.env
python -m uvicorn app:app --host 0.0.0.0 --port 8081

# Frontend (different terminal)
cd pumpfun_frontend
npm install
npm run dev  # http://127.0.0.1:3001
```

Set `NEXT_PUBLIC_PUMPFUN_API_URL` if the frontend should point at a different API host.

## Repository Layout

```
pumpfun_train/     # training + CLI + model artifacts
pumpfun_api/       # FastAPI prediction service
pumpfun_frontend/  # Next.js UI
docs/              # pump.fun docs and workflows
```

## Documentation

- `docs/README.md` for the pump.fun doc index
- `docs/pumpfun-process.md` for the mint-level prediction walkthrough
- `docs/training-workflow.md` for the training workflow
- `docs/api.md` for the pump.fun API surface
- `docs/environment.md` for environment variables
- `docs/frontend.md` for frontend setup
