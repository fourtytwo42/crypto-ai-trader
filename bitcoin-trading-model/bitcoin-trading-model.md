---
title: Bitcoin Trading Model
status: planning
category: AI / Machine Learning
tags: [ai, ml, trading, bitcoin, time-series, forecasting, patchtst, postgresql, python]
keywords: [patchtst, neuralforecast, postgresql, fastapi, python, time-series, forecasting, backtesting]
created: 2025-01-28
---

# Bitcoin Trading Model

**AI-powered Bitcoin price direction prediction system** - A Python-based machine learning system that trains time-series forecasting models (PatchTST/NHITS) on Bitcoin daily candle data to predict price direction with improved odds. Features PostgreSQL data staging, terminal menu interface, REST API for predictions, CLI argument support for automation, and comprehensive backtesting capabilities.

## Concept

**Problem Statement:** Predicting Bitcoin price direction is difficult due to market noise, non-stationarity, and the need to account for transaction costs. Most naive approaches (predicting every move) result in coin-flip odds after fees. We need a system that only makes predictions when there's sufficient edge to overcome costs.

**Solution Vision:** Train a time-series forecasting model (PatchTST or NHITS) on 10+ years of daily Bitcoin OHLCV data from Kraken. The model predicts forward returns (not raw price) with uncertainty estimates. A decision layer converts predictions into directional trades only when confidence exceeds a threshold, accounting for fees and slippage. The system includes data staging in PostgreSQL, model training pipeline, terminal interface for interactive use, REST API for programmatic access, and walk-forward backtesting with proper leakage controls.

**Core Philosophy:**
- **Scale-free features:** Use log returns, not raw prices, so model generalizes across price levels
- **No-trade zones:** Only predict when edge is strong enough to overcome costs
- **Proper evaluation:** Walk-forward testing with purged/embargo splits to prevent leakage
- **Uncertainty-aware:** Predict quantiles or distributions, not just point estimates
- **Simple first:** Start with daily candles (1 model), add more timeframes later if v1 works

**Target Users:**
- Individual traders wanting to improve directional prediction odds
- Developers building trading systems that need ML predictions
- Researchers studying time-series forecasting on financial data

**Success Metrics:**
- Win rate on traded signals (target: >55% after costs)
- Sharpe ratio of backtested strategy
- Model performance vs. baselines (random walk, ARIMA, simple ML)
- Prediction accuracy on walk-forward test sets

## Core Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Terminal Menu Interface                    │
│  Train | Predict | Backtest | View Data | Model Info | Exit  │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                    CLI Argument Parser                        │
│  --train | --predict | --backtest | --api | --menu          │
└───────┬───────────────────────────────────────────┬─────────┘
        │                                           │
┌───────▼────────┐                      ┌──────────▼──────────┐
│  PostgreSQL    │                      │  FastAPI Server     │
│  (Candle Data) │                      │  (REST API)          │
│  - candles     │                      │  /predict            │
│  - features    │                      │  /health             │
│  - predictions │                      │  /models             │
└───────┬────────┘                      └──────────┬──────────┘
        │                                           │
        │                      ┌────────────────────┼──────────┐
        │                      │                    │          │
┌───────▼────┐  ┌─────────────▼─────┐  ┌──────────▼─────┐  ┌─▼──────────┐
│  Data      │  │  Model Training   │  │  Prediction     │  │ Backtest   │
│  Pipeline  │  │  Pipeline         │  │  Engine        │  │ Engine     │
│  - CSV     │  │  - Feature Eng.   │  │  - Load Model   │  │ - Walk-    │
│  - Load    │  │  - Train/Val/Test│  │  - Predict      │  │   Forward  │
│  - Feature │  │  - Save Weights   │  │  - Convert to    │  │ - Metrics  │
│  Extract   │  │  - Save Metadata │  │    Signals      │  │ - P&L     │
└────────────┘  └───────────────────┘  └─────────────────┘  └───────────┘
```

**Key Components:**

1. **PostgreSQL Database** - Stores raw candles, engineered features, model metadata, predictions, backtest results
2. **Data Pipeline** - Loads Kraken CSV files, extracts features (log returns, range, volume), normalizes data
3. **Model Training Pipeline** - Trains PatchTST or NHITS models using NeuralForecast, saves weights and scalers
4. **Prediction Engine** - Loads trained models, generates forecasts with uncertainty, converts to directional signals
5. **Backtest Engine** - Walk-forward evaluation with proper splits, calculates P&L with fees/slippage, generates metrics
6. **Terminal Menu** - Interactive interface for all operations with rich terminal output
7. **CLI Arguments** - Non-interactive mode for automation (AI agents, scripts, cron jobs)
8. **FastAPI Server** - REST API for programmatic access to predictions

## Status

**Current Phase:** Planning

**Next Steps:**
- [ ] Create database schema and migration scripts
- [ ] Implement CSV data loader for Kraken format
- [ ] Build feature engineering pipeline
- [ ] Set up model training pipeline with NeuralForecast
- [ ] Create terminal menu interface
- [ ] Implement CLI argument parser
- [ ] Build FastAPI server
- [ ] Implement backtesting engine
- [ ] Add comprehensive logging and terminal output

## Project Artifacts

**Required Artifacts:**
- [Implementation Guide](bitcoin-trading-model-implementation.md) - Setup, structure, patterns, deployment (REQUIRED)
- [Architecture Details](bitcoin-trading-model-architecture.md) - Complete system design, component breakdown, data flow (REQUIRED)
- [Database Schema](bitcoin-trading-model-database.md) - Complete PostgreSQL schema, tables, relationships, indexes (REQUIRED)
- [API Specifications](bitcoin-trading-model-api.md) - All API endpoints, request/response formats, error handling (REQUIRED)

**Future Artifacts (as needed):**
- Model Evaluation Guide - Detailed evaluation methodology, metrics, baseline comparisons
- Backtesting Methodology - Walk-forward procedure, leakage prevention, cost modeling
- Production Deployment Guide - Deployment strategy, monitoring, model retraining schedule

## Technology Stack

**DECISION: All technology choices are explicit with versions and rationale to eliminate ambiguity.**

- **Language:** Python 3.11.7 (exact version)
  - **Rationale:** Latest stable 3.11.x release, type hints support, performance improvements
  - **Why not 3.12:** 3.11 is more stable, better library compatibility

- **ML Framework:** NeuralForecast 1.7.0 (exact version)
  - **Rationale:** State-of-the-art time-series models, unified API, built-in probabilistic forecasting
  - **Models:** PatchTST (primary), NHITS (baseline comparison), TSMixer (alternative)
  - **Why NeuralForecast:** Modern, actively maintained, supports quantile forecasting out of the box
  - **Why not Darts:** Less modern models, more complex API
  - **Why not PyTorch Forecasting:** TFT-focused, less flexible

- **Database:** PostgreSQL 15.4 (exact version)
  - **Rationale:** Relational structure needed, easy querying, reliable
  - **Why not TimescaleDB:** Overkill for daily candles, adds complexity, can add later if needed
  - **Why not SQLite:** Too limited for production, concurrency issues
  - **Why not InfluxDB:** Overkill, less queryable for our use case

- **ORM:** SQLAlchemy 2.0.23 (exact version)
  - **Rationale:** Industry standard, type-safe, async support
  - **Why not Prisma:** Python support is limited, SQLAlchemy is more mature

- **Migrations:** Alembic 1.13.1 (exact version)
  - **Rationale:** Standard for SQLAlchemy, reliable migrations

- **API Framework:** FastAPI 0.109.0 (exact version)
  - **Rationale:** Modern async framework, automatic OpenAPI docs, type validation
  - **Why not Flask:** Older, less type-safe, no async by default
  - **Why not Django REST:** Too heavy for simple API

- **CLI Framework:** Click 8.1.7 (exact version)
  - **Rationale:** Better than argparse for complex CLIs, decorator-based, better help generation
  - **Why not argparse:** More verbose, less intuitive
  - **Why not Typer:** Newer, less stable, Click is battle-tested

- **Terminal UI:** Rich 13.7.0 (exact version)
  - **Rationale:** Beautiful terminal output, tables, progress bars, cross-platform
  - **Why not curses:** Too complex, Rich is easier to use

- **Data Processing:** 
  - pandas 2.1.4 (exact version)
  - numpy 1.26.3 (exact version)
  - **Rationale:** Industry standard, well-tested, performant

- **Model Storage:** 
  - PyTorch 2.1.2 (exact version) - model weights (.pt files)
  - JSON - model metadata
  - pickle - scalers (.pkl files)
  - **Rationale:** Standard PyTorch format, human-readable metadata, efficient scaler storage

- **Testing:** 
  - pytest 7.4.4 (exact version)
  - pytest-cov 4.1.0 (exact version)
  - pytest-asyncio 0.23.3 (exact version) - for async tests
  - **Rationale:** Industry standard, excellent coverage tools

- **Type Checking:** mypy 1.8.0 (exact version)
  - **Rationale:** Best Python type checker, strict mode support

- **Logging:** structlog 24.1.0 (exact version)
  - **Rationale:** Structured logging, better than standard logging for production

## Data Source

- **Initial Data:** Kraken BTCUSD_Daily_OHLC.csv (2013-10-06 to 2023-12-31, ~3,727 daily candles)
- **Format:** timestamp, open, high, low, close, volume, trades
- **Future:** Can extend to pull from exchange APIs (Binance, Coinbase, etc.)

## Model Configuration (v1)

**DECISION: All model choices are explicit with specific values, no ambiguity.**

- **Timeframe:** Daily candles (1 day = 24 hours)
  - **Rationale:** Less noise than hourly, sufficient data (3,727 days), easier to start
  - **Future:** Can add 4h, 1h, 5m models later if daily works

- **Horizon:** Next 1 day (predict tomorrow's return)
  - **Rationale:** Short horizon reduces compounding error, more actionable
  - **Future:** Can extend to 3-7 days for swing trading

- **Primary Model:** PatchTST (from NeuralForecast)
  - **Rationale:** State-of-the-art transformer for time-series, efficient attention via patching
  - **Configuration:**
    - Hidden size: 512 (exact)
    - Layers: 6 (exact)
    - Patch length: 16 (exact - ~2 weeks per patch for daily data)
    - Stride: 8 (exact)
    - Context length: 128 days (exact - ~4 months)
    - Parameters: ~15-25M (estimated)
  - **Why PatchTST over NHITS:** Better for longer context, attention mechanism helps with patterns

- **Baseline Model:** NHITS (from NeuralForecast)
  - **Rationale:** Strong baseline, faster training, hierarchical interpolation
  - **Configuration:**
    - Hidden size: 512 (exact)
    - Layers: 3 (exact - hierarchical)
    - Context length: 128 days (exact)
    - Parameters: ~10-15M (estimated)
  - **Why include:** Compare against PatchTST, verify PatchTST is better

- **Features (Exact List):**
  1. `return` = log(C_t / C_{t-1}) - log return
  2. `range` = log(H_t / L_t) - range/volatility proxy
  3. `body` = log(C_t / O_t) - body size
  4. `dlog_volume` = log(V_t) - log(V_{t-1}) - volume change
  5. `ret_mean_7` = 7-day rolling mean of returns
  6. `ret_std_7` = 7-day rolling std of returns
  7. `ret_mean_30` = 30-day rolling mean of returns
  8. `ret_std_30` = 30-day rolling std of returns
  - **Total:** 8 features (exact count)
  - **Rationale:** Scale-free, stationary-ish, captures trend and volatility

- **Normalization:** Rolling z-score (30-day window)
  - **Formula:** z_t = (x_t - mean(x_{t-30..t-1})) / std(x_{t-30..t-1})
  - **Rationale:** Prevents leakage (only uses past), handles non-stationarity
  - **Why not min-max:** Sensitive to outliers, doesn't handle non-stationarity well

- **Target:** Forward log return with quantiles [0.1, 0.5, 0.9] (exact values)
  - **Rationale:** Provides uncertainty estimates, enables better risk management
  - **Why quantiles:** Standard in financial ML, interpretable, NeuralForecast supports it

- **Signal Threshold:** τ = 0.005 (0.5%) (exact value)
  - **Rationale:** Accounts for typical exchange fees (0.1%) + slippage (0.05%) + buffer
  - **Decision Rule:**
    - Long if: q50 > 0.005 AND q10 > -0.005 (upside bias, limited downside)
    - Short if: q50 < -0.005 AND q90 < 0.005 (downside bias, limited upside)
    - Flat otherwise
  - **Why this rule:** Only trades when edge is strong enough to overcome costs

## Decisions Made

**CRITICAL: All decisions are explicit with rationale. No ambiguity allowed.**

### Model Architecture Decision

**Decision:** Use PatchTST as primary model, NHITS as baseline comparison.

**Rationale:**
- PatchTST uses efficient attention via patching (reduces O(n²) to O(n))
- Better for longer context windows (128 days)
- State-of-the-art performance on time-series benchmarks
- NHITS is faster but less capable with long context
- Both available in NeuralForecast with unified API

**Alternatives Considered:**
- LSTM/GRU: Older, less efficient, worse performance
- Pure Transformer: Too expensive (O(n²) attention)
- TSMixer: Simpler but less powerful than PatchTST

### Database Decision

**Decision:** Use PostgreSQL 15.4 (plain, not TimescaleDB).

**Rationale:**
- Daily candles don't need time-series optimization (TimescaleDB overkill)
- PostgreSQL JSONB perfect for model metadata
- Can add TimescaleDB later if needed for higher frequency
- Simpler setup, fewer dependencies
- Relational structure needed for features, predictions, backtests

**Alternatives Considered:**
- TimescaleDB: Overkill for daily data, adds complexity
- SQLite: Too limited, concurrency issues
- InfluxDB: Overkill, less queryable

### CLI Framework Decision

**Decision:** Use Click 8.1.7 (not argparse, not Typer).

**Rationale:**
- Decorator-based syntax is cleaner than argparse
- Better help generation
- Battle-tested, stable
- Typer is newer but less stable
- argparse is too verbose

### Feature Engineering Decision

**Decision:** Use exactly 8 scale-free features (log returns, ratios, rolling stats).

**Rationale:**
- Scale-free features ensure model works at any price level ($10k or $200k)
- Log returns are standard in financial ML
- Rolling stats capture trend and volatility
- 8 features is sufficient (more would risk overfitting with 3,727 samples)

**Exact Features:**
1. return = log(C_t / C_{t-1})
2. range = log(H_t / L_t)
3. body = log(C_t / O_t)
4. dlog_volume = log(V_t) - log(V_{t-1})
5. ret_mean_7 = 7-day rolling mean
6. ret_std_7 = 7-day rolling std
7. ret_mean_30 = 30-day rolling mean
8. ret_std_30 = 30-day rolling std

### Normalization Decision

**Decision:** Use rolling z-score with 30-day window (not min-max, not train-set stats).

**Rationale:**
- Rolling z-score prevents leakage (only uses past data)
- Handles non-stationarity (markets change over time)
- Min-max is sensitive to outliers
- Train-set stats would leak future information

### Signal Threshold Decision

**Decision:** Use τ = 0.005 (0.5%) as signal threshold.

**Rationale:**
- Accounts for fees (0.1%) + slippage (0.05%) + buffer (0.35%)
- Only trades when edge is strong enough to overcome costs
- Prevents trading on tiny random moves
- Can be tuned based on backtest results

### Backtesting Decision

**Decision:** Use walk-forward evaluation with purged/embargo splits.

**Rationale:**
- Walk-forward simulates real-world usage (train on past, test on future)
- Purged splits prevent leakage from overlapping horizons
- Embargo periods add extra safety
- Standard practice in financial ML (mlfinlab methodology)

**Exact Procedure:**
1. Train on 2013-2019, validate on 2020-2021, test on 2022-2023
2. Purge 1 day between train/test (prevents overlap)
3. Roll forward: train on 2014-2020, validate on 2021-2022, test on 2023
4. Repeat until data exhausted

### API Framework Decision

**Decision:** Use FastAPI 0.109.0 (not Flask, not Django REST).

**Rationale:**
- Modern async framework
- Automatic OpenAPI docs
- Type validation with Pydantic
- Better performance than Flask
- Lighter than Django REST

### Terminal UI Decision

**Decision:** Use Rich 13.7.0 for terminal output (not curses, not plain print).

**Rationale:**
- Beautiful tables, progress bars, colors
- Cross-platform
- Easy to use
- Better than plain print statements
- Simpler than curses

## Related Projects

- None currently (standalone project)

