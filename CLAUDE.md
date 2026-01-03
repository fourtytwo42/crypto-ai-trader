# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bitcoin Trading Model - An AI-powered Bitcoin price direction prediction system using time-series forecasting models (PatchTST/NHITS) trained on daily candle data. Features PostgreSQL data staging, terminal menu interface, REST API for predictions, and walk-forward backtesting.

**Current Status:** Planning phase - core infrastructure implemented, model training and prediction pipelines in development.

## Build and Development Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run application (CLI with terminal menu)
python -m src.main
# or use the btm entry point after pip install -e .
btm

# Run specific CLI commands
python -m src.main --menu          # Interactive menu (default)
python -m src.main --train         # Train model
python -m src.main --predict       # Generate prediction
python -m src.main --backtest      # Run backtest
python -m src.main --api           # Start FastAPI server

# Testing
pytest                              # Run all tests
pytest tests/test_training.py      # Run single test file
pytest -k "test_feature"           # Run tests matching pattern
pytest --cov=src --cov-report=html # Coverage report

# Code quality
ruff check src tests               # Linting
black src tests                    # Formatting
mypy src                           # Type checking

# Database migrations
alembic upgrade head               # Apply migrations
alembic revision --autogenerate -m "description"  # Create migration
```

## Architecture

### Source Layout (`src/`)

```
src/
├── main.py              # Entry point, structlog configuration
├── config.py            # Pydantic settings from environment
├── data/                # Data pipeline
│   ├── csv_loader.py    # Kraken CSV format loader
│   ├── feature_extractor.py  # Log returns, range, volume features
│   ├── normalizer.py    # Rolling z-score normalization
│   ├── kucoin_client.py # Exchange API client
│   └── pipeline.py      # Orchestrates data flow
├── database/            # PostgreSQL layer
│   ├── connection.py    # SQLAlchemy async engine
│   ├── models.py        # ORM models (candles, features, predictions)
│   └── operations.py    # CRUD operations
├── training/            # Model training
│   ├── trainer.py       # Training orchestration
│   ├── data_preparation.py  # Train/val/test splits
│   ├── model_factory.py # Creates PatchTST/NHITS models
│   └── evaluator.py     # Model evaluation metrics
├── prediction/          # Inference
│   ├── model_loader.py  # Load saved models
│   ├── predictor.py     # Generate forecasts with quantiles
│   └── signal_generator.py  # Convert predictions to long/short/flat
├── backtest/            # Evaluation
│   ├── walk_forward.py  # Walk-forward evaluation
│   ├── signal_executor.py   # Simulates trading with fees
│   └── metrics_calculator.py  # Sharpe, win rate, drawdown
├── cli/                 # Terminal interface
│   ├── cli.py           # Click argument parser
│   ├── menu.py          # Rich terminal menu
│   └── commands.py      # Command implementations
└── api/                 # REST API
    ├── main.py          # FastAPI app
    ├── routes.py        # Endpoints (/predict, /health, /models)
    ├── models.py        # Pydantic request/response
    └── dependencies.py  # FastAPI dependencies
```

### Key Data Flow

1. **Training Flow:** CSV -> Load -> Extract Features (8 scale-free features) -> Normalize (rolling z-score) -> Train/Val/Test Split -> Train Model -> Save Weights + Metadata
2. **Prediction Flow:** Latest Data -> Extract Features -> Normalize -> Load Model -> Predict Quantiles [0.1, 0.5, 0.9] -> Generate Signal (long/short/flat based on threshold)
3. **Backtest Flow:** Walk-forward windows -> Train on Window A -> Test on Window B -> Execute Signals with Fees -> Calculate Metrics -> Roll Forward

### Feature Engineering (Exact 8 Features)

1. `return` = log(C_t / C_{t-1}) - log return
2. `range` = log(H_t / L_t) - volatility proxy
3. `body` = log(C_t / O_t) - body size
4. `dlog_volume` = log(V_t) - log(V_{t-1}) - volume change
5. `ret_mean_7` - 7-day rolling mean of returns
6. `ret_std_7` - 7-day rolling std of returns
7. `ret_mean_30` - 30-day rolling mean of returns
8. `ret_std_30` - 30-day rolling std of returns

All features are scale-free (work at any BTC price level).

## Technology Stack (Pinned Versions)

- **Python 3.11.7** (exact)
- **NeuralForecast 1.7.0** - PatchTST/NHITS models
- **PyTorch 2.1.2** - Model backend
- **PostgreSQL 15.4** - Data storage
- **SQLAlchemy 2.0.23** + Alembic - ORM and migrations
- **FastAPI 0.109.0** - REST API
- **Click 8.1.7** - CLI framework
- **Rich 13.7.0** - Terminal UI
- **structlog 24.1.0** - Structured logging
- **pytest 7.4.4** - Testing (90%+ coverage target)
- **mypy 1.8.0** (strict mode), **ruff**, **black** - Code quality

## Configuration

All settings via environment variables (see `.env.example`). Key settings:

- `DATABASE_URL` - PostgreSQL connection string
- `MODEL_TYPE` - "patchtst" or "nhits"
- `MODEL_CONTEXT_LENGTH` - 128 days default
- `SIGNAL_THRESHOLD` - 0.005 (0.5%) default
- `TRAIN_DEVICE` - "cuda" or "cpu"

## Model Details

- **Primary:** PatchTST (hidden=512, layers=6, patch=16, context=128 days)
- **Baseline:** NHITS for comparison
- **Output:** Quantiles [0.1, 0.5, 0.9] for uncertainty estimation
- **Signal Logic:** Long if q50 > threshold AND q10 > -threshold; Short if opposite; Flat otherwise

## Testing Requirements

- 90%+ test coverage required
- Tests in `tests/` directory
- Use pytest with async support (`pytest-asyncio`)
- Coverage config in `pyproject.toml`

## Documentation

- `bitcoin-trading-model/` - Project design documents
- `docs/postgres-setup.md` - Database setup
- `docs/training-workflow.md` - Training procedures
- `docs/kucoin-backfill.md` - Data backfill process
