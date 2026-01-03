# Bitcoin Trading Model - Implementation Guide

**Complete implementation guide, development setup, project structure, coding patterns, deployment strategy, and testing requirements.**

## Table of Contents

1. [Project Structure](#project-structure)
2. [Development Setup](#development-setup)
3. [Database Setup](#database-setup)
4. [Environment Variables](#environment-variables)
5. [Development Phases](#development-phases)
6. [Coding Patterns & Conventions](#coding-patterns--conventions)
7. [Data Pipeline Implementation](#data-pipeline-implementation)
8. [Model Training Implementation](#model-training-implementation)
9. [Prediction Implementation](#prediction-implementation)
10. [Backtesting Implementation](#backtesting-implementation)
11. [Terminal Interface Implementation](#terminal-interface-implementation)
12. [CLI Implementation](#cli-implementation)
13. [API Implementation](#api-implementation)
14. [Testing Strategy](#testing-strategy)
15. [Deployment Strategy](#deployment-strategy)

## Project Structure

```
bitcoin-trading-model/
├── .env.example                  # Example environment variables
├── .gitignore
├── README.md                      # Project overview and quick start
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata and tool configs
├── setup.py                       # Package installation
├── alembic.ini                    # Alembic config for migrations
├── alembic/                       # Database migrations
│   ├── versions/
│   └── env.py
├── data/                          # Data files (gitignored)
│   └── BTCUSD_Daily_OHLC.csv     # Kraken CSV file
├── models/                        # Saved model files (gitignored)
│   ├── model_v1.pt
│   ├── model_v1_metadata.json
│   └── scaler_v1.pkl
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point (CLI + menu)
│   ├── config.py                  # Configuration management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py          # DB connection pool
│   │   ├── models.py              # SQLAlchemy models
│   │   └── operations.py          # DB operations
│   ├── data/
│   │   ├── __init__.py
│   │   ├── csv_loader.py          # Load Kraken CSV
│   │   ├── feature_extractor.py   # Feature engineering
│   │   ├── normalizer.py          # Normalization
│   │   └── pipeline.py            # End-to-end data pipeline
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Main training orchestration
│   │   ├── data_preparation.py    # Train/val/test splits
│   │   ├── model_factory.py       # Create models (PatchTST, NHITS)
│   │   ├── evaluator.py           # Model evaluation
│   │   └── config.py              # Training configs
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── model_loader.py         # Load saved models
│   │   ├── predictor.py           # Generate predictions
│   │   └── signal_generator.py   # Convert to signals
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── walk_forward.py         # Walk-forward evaluation
│   │   ├── signal_executor.py     # Simulate trading
│   │   └── metrics_calculator.py  # Calculate metrics
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── cli.py                  # Argument parser
│   │   ├── commands.py             # Command implementations
│   │   ├── menu.py                 # Main menu
│   │   ├── train_menu.py           # Training menu
│   │   ├── predict_menu.py        # Prediction menu
│   │   ├── backtest_menu.py       # Backtest menu
│   │   └── data_menu.py            # Data viewing menu
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app
│       ├── routes.py               # API endpoints
│       ├── models.py               # Pydantic models
│       └── dependencies.py         # API dependencies
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_data/
│   │   └── sample_candles.csv      # Test data
│   ├── test_data_pipeline.py
│   ├── test_training.py
│   ├── test_prediction.py
│   ├── test_backtest.py
│   ├── test_cli.py
│   └── test_api.py
└── docs/
    ├── README.md                   # Documentation index
    ├── SETUP.md                    # Detailed setup guide
    ├── USAGE.md                    # Usage examples
    └── API.md                      # API documentation
```

## Development Setup

### Prerequisites

**EXACT VERSIONS REQUIRED (no ambiguity):**

- **Python:** 3.11.7 (exact version required)
  - **Check:** `python --version` must show 3.11.7
  - **Why:** Type hints, performance, library compatibility

- **PostgreSQL:** 15.4 or higher (exact minimum: 15.4)
  - **Check:** `psql --version` must show 15.4+
  - **Why:** JSONB support, performance improvements

- **Git:** 2.42.0 or higher (for version control)
  - **Check:** `git --version`

- **CUDA:** 11.8 or 12.1 (if using GPU)
  - **Check:** `nvidia-smi` for GPU availability
  - **RTX 3090:** 24GB VRAM, supports training models up to ~50M parameters
  - **CPU Fallback:** All code must work on CPU (slower but functional)

### Step 1: Clone and Setup Environment

**EXACT COMMANDS (no ambiguity):**

```bash
# Create virtual environment (exact command)
python3.11 -m venv venv

# Activate virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Verify Python version (must be 3.11.7)
python --version
# Expected output: Python 3.11.7

# Upgrade pip (exact version)
pip install --upgrade pip==24.0

# Install dependencies (exact versions from requirements.txt)
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "import neuralforecast; print(neuralforecast.__version__)"
# Should print: 1.7.0
```

### Step 1a: Create requirements.txt

**EXACT VERSIONS REQUIRED (no ambiguity, no ranges):**

Create `requirements.txt` with these exact versions:

```txt
# Core ML Framework
neuralforecast==1.7.0
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2

# Data Processing
pandas==2.1.4
numpy==1.26.3

# Database
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9

# API
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# CLI
click==8.1.7

# Terminal UI
rich==13.7.0

# Logging
structlog==24.1.0

# Testing
pytest==7.4.4
pytest-cov==4.1.0
pytest-asyncio==0.23.3

# Type Checking
mypy==1.8.0
types-requests==2.31.0.10

# Development
black==24.1.0
ruff==0.1.11
```

**Rationale for each:**
- **neuralforecast 1.7.0:** Latest stable with PatchTST/NHITS support
- **torch 2.1.2:** Compatible with CUDA 11.8/12.1, stable
- **pandas 2.1.4:** Latest 2.1.x (stable), type hints support
- **numpy 1.26.3:** Compatible with pandas 2.1.4
- **sqlalchemy 2.0.23:** Modern async support, type-safe
- **fastapi 0.109.0:** Latest stable, OpenAPI 3.1 support
- **click 8.1.7:** Battle-tested, better than argparse
- **rich 13.7.0:** Beautiful terminal output
- **structlog 24.1.0:** Structured logging for production

### Step 2: Install PostgreSQL

**Windows:**
- Download from https://www.postgresql.org/download/windows/
- Install with default settings
- Note password for postgres user

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

**Mac:**
```bash
brew install postgresql
brew services start postgresql
```

### Step 3: Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE bitcoin_trading;
CREATE USER trading_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE bitcoin_trading TO trading_user;
\q
```

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# DATABASE_URL=postgresql://trading_user:your_password@localhost:5432/bitcoin_trading
```

### Step 5: Run Migrations

```bash
# Initialize Alembic (if not already done)
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head
```

### Step 6: Load Initial Data

```bash
# Place BTCUSD_Daily_OHLC.csv in data/ directory
# Then run data loader
python -m src.cli --load-data data/BTCUSD_Daily_OHLC.csv
```

### Step 7: Verify Setup

```bash
# Run tests
pytest

# Start interactive menu
python -m src.cli --menu
```

## Database Setup

See [Database Schema](bitcoin-trading-model-database.md) for complete schema details.

### Quick Setup Commands

```bash
# Create database (if not using migrations)
psql -U postgres -c "CREATE DATABASE bitcoin_trading;"

# Run migrations
alembic upgrade head

# Verify tables created
psql -U postgres -d bitcoin_trading -c "\dt"
```

## Environment Variables

**EXACT FORMAT REQUIRED (no ambiguity):**

Create `.env` file in project root with these exact variables:

```bash
# Database (REQUIRED - exact format)
DATABASE_URL=postgresql://trading_user:password@localhost:5432/bitcoin_trading
# Format: postgresql://[user]:[password]@[host]:[port]/[database]
# Default port: 5432

# Model Storage (REQUIRED - exact paths)
MODEL_DIR=models
# Directory for saved models (relative to project root)
# Will contain: .pt files, _metadata.json files, _scaler.pkl files

# Data Storage (REQUIRED - exact paths)
DATA_DIR=data
# Directory for CSV files (relative to project root)
# Place BTCUSD_Daily_OHLC.csv here

# API Configuration (REQUIRED - exact values)
API_HOST=0.0.0.0
# Bind to all interfaces (0.0.0.0) or localhost only (127.0.0.1)
API_PORT=8000
# Port number (must be integer, 1024-65535)
API_RELOAD=true
# Auto-reload on code changes (true/false, development only)

# Training Configuration (REQUIRED - exact values)
TRAIN_DEVICE=cuda
# Options: "cuda" (GPU) or "cpu" (CPU fallback)
# Must match available hardware
TRAIN_BATCH_SIZE=32
# Batch size (must be integer, 8-128 recommended)
TRAIN_LEARNING_RATE=0.0001
# Learning rate (must be float, 0.00001-0.001 range)
TRAIN_EPOCHS=100
# Maximum epochs (must be integer, 50-200 recommended)
TRAIN_EARLY_STOPPING_PATIENCE=10
# Early stopping patience (must be integer, 5-20 recommended)
TRAIN_VALIDATION_SPLIT=0.15
# Validation split ratio (must be float, 0.1-0.2 range)

# Model Configuration (REQUIRED - exact values)
MODEL_TYPE=patchtst
# Options: "patchtst" (primary) or "nhits" (baseline)
MODEL_CONTEXT_LENGTH=128
# Context length in days (must be integer, 64-256 range)
MODEL_HORIZON=1
# Prediction horizon in days (must be integer, 1-7 range)
MODEL_HIDDEN_SIZE=512
# Hidden size (must be integer, 256-1024 range)
MODEL_NUM_LAYERS=6
# Number of layers (must be integer, 3-12 range)
MODEL_PATCH_LENGTH=16
# Patch length for PatchTST (must be integer, 8-32 range)
MODEL_STRIDE=8
# Stride for PatchTST (must be integer, 4-16 range)

# Signal Generation (REQUIRED - exact values)
SIGNAL_THRESHOLD=0.005
# Threshold for signal generation (must be float, 0.001-0.01 range)
# 0.005 = 0.5% (accounts for fees + slippage + buffer)

# Backtesting Configuration (REQUIRED - exact values)
BACKTEST_FEES=0.001
# Fee rate per trade (must be float, 0.0001-0.01 range)
# 0.001 = 0.1% (typical exchange fee)
BACKTEST_SLIPPAGE=0.0005
# Slippage rate per trade (must be float, 0.0001-0.005 range)
# 0.0005 = 0.05% (typical slippage)
BACKTEST_TRAIN_WINDOW=1825
# Training window in days (must be integer, 365-3650 range)
# 1825 = 5 years
BACKTEST_TEST_WINDOW=365
# Test window in days (must be integer, 30-730 range)
# 365 = 1 year

# Walk-Forward Configuration (REQUIRED - exact values)
WALK_FORWARD_PURGE_DAYS=1
# Days to purge between train/test (must be integer, 0-7 range)
# Prevents leakage from overlapping horizons
WALK_FORWARD_EMBARGO_DAYS=0
# Days to embargo after test period (must be integer, 0-7 range)
# Additional leakage prevention

# Logging (REQUIRED - exact values)
LOG_LEVEL=INFO
# Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE=logs/app.log
# Log file path (relative to project root)
LOG_FORMAT=json
# Options: "json" (structured) or "text" (human-readable)
```

## Development Phases

### Phase 1: Data Pipeline (Week 1)

**Goals:**
- Load Kraken CSV files
- Extract features
- Store in PostgreSQL
- Basic data validation

**Deliverables:**
- CSV loader working
- Feature extraction pipeline
- Database schema and migrations
- Data validation tests

### Phase 2: Model Training (Week 2)

**Goals:**
- Set up NeuralForecast
- Implement PatchTST training
- Train/val/test splits
- Model saving/loading

**Deliverables:**
- Training pipeline working
- Model can be trained and saved
- Basic evaluation metrics

### Phase 3: Prediction & Signals (Week 3)

**Goals:**
- Load trained models
- Generate predictions
- Convert to signals
- Terminal output for predictions

**Deliverables:**
- Prediction engine working
- Signal generation with thresholds
- Terminal interface for predictions

### Phase 4: Backtesting (Week 4)

**Goals:**
- Walk-forward evaluation
- Signal execution simulation
- Metrics calculation
- Backtest results display

**Deliverables:**
- Backtesting engine working
- Proper leakage prevention
- Comprehensive metrics

### Phase 5: Terminal Interface (Week 5)

**Goals:**
- Interactive menu system
- Rich terminal output
- All operations accessible via menu

**Deliverables:**
- Complete terminal menu
- Beautiful terminal UI
- All features accessible

### Phase 6: CLI & API (Week 6)

**Goals:**
- CLI argument parser
- FastAPI server
- API endpoints for predictions

**Deliverables:**
- CLI working for automation
- API server running
- API documentation

## Coding Patterns & Conventions

### Python Style

- **PEP 8:** Follow Python style guide
- **Type Hints:** Use type hints for all functions
- **Docstrings:** Google-style docstrings for all public functions
- **Line Length:** 100 characters max

### Code Organization

- **Modules:** One class/concern per file
- **Functions:** Small, focused functions (< 50 lines)
- **Classes:** Use classes for stateful operations (database, models)
- **Constants:** Define constants at module level

### Error Handling

- **Exceptions:** Use specific exceptions, not bare `except`
- **Logging:** Use Python logging module, not print statements
- **Validation:** Validate inputs at function boundaries
- **Graceful Degradation:** Handle errors gracefully, don't crash

### Example Code Structure

**EXACT PATTERNS REQUIRED (no ambiguity):**

```python
"""Module docstring explaining purpose.

This module handles feature extraction from OHLCV candle data.
All features are scale-free (log returns, ratios) to ensure
model generalizes across price levels.
"""

from typing import Optional, Tuple
import logging
import pandas as pd
import numpy as np

# Use structlog for structured logging (not standard logging)
import structlog

logger = structlog.get_logger(__name__)

# Constants (exact values, no magic numbers)
DEFAULT_THRESHOLD: float = 0.005  # 0.5% threshold
ROLLING_WINDOW_7: int = 7  # 7-day rolling window
ROLLING_WINDOW_30: int = 30  # 30-day rolling window

# Required columns (exact list)
REQUIRED_COLUMNS: list[str] = ['open', 'high', 'low', 'close', 'volume']

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract scale-free features from candle data.
    
    Extracts exactly 8 features:
    1. return: log(C_t / C_{t-1})
    2. range: log(H_t / L_t)
    3. body: log(C_t / O_t)
    4. dlog_volume: log(V_t) - log(V_{t-1})
    5. ret_mean_7: 7-day rolling mean of returns
    6. ret_std_7: 7-day rolling std of returns
    7. ret_mean_30: 30-day rolling mean of returns
    8. ret_std_30: 30-day rolling std of returns
    
    Args:
        df: DataFrame with OHLCV columns. Must have columns:
            'open', 'high', 'low', 'close', 'volume'
        
    Returns:
        DataFrame with extracted features. Original columns preserved.
        Features added: 'return', 'range', 'body', 'dlog_volume',
        'ret_mean_7', 'ret_std_7', 'ret_mean_30', 'ret_std_30'
        
    Raises:
        ValueError: If required columns missing
        ValueError: If data is empty
        ValueError: If price consistency checks fail (high < low, etc.)
    """
    # Validation (exact checks, no ambiguity)
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Price consistency checks (exact validation)
    if (df['high'] < df['low']).any():
        raise ValueError("Invalid data: high < low detected")
    if (df['close'] < df['low']).any() or (df['close'] > df['high']).any():
        raise ValueError("Invalid data: close outside [low, high] range")
    
    logger.info("Extracting features", num_rows=len(df))
    
    # Feature extraction (exact formulas)
    result = df.copy()
    
    # 1. Log return (exact formula)
    result['return'] = np.log(result['close'] / result['close'].shift(1))
    
    # 2. Range (exact formula)
    result['range'] = np.log(result['high'] / result['low'])
    
    # 3. Body (exact formula)
    result['body'] = np.log(result['close'] / result['open'])
    
    # 4. Volume change (exact formula)
    result['dlog_volume'] = np.log(result['volume']) - np.log(result['volume'].shift(1))
    
    # 5-6. 7-day rolling stats (exact windows)
    result['ret_mean_7'] = result['return'].rolling(window=ROLLING_WINDOW_7).mean()
    result['ret_std_7'] = result['return'].rolling(window=ROLLING_WINDOW_7).std()
    
    # 7-8. 30-day rolling stats (exact windows)
    result['ret_mean_30'] = result['return'].rolling(window=ROLLING_WINDOW_30).mean()
    result['ret_std_30'] = result['return'].rolling(window=ROLLING_WINDOW_30).std()
    
    # Remove rows with NaN (from rolling windows and shifts)
    result = result.dropna().reset_index(drop=True)
    
    logger.info("Features extracted", num_rows=len(result), num_features=8)
    
    return result
```

## Data Pipeline Implementation

### CSV Loader

```python
# src/data/csv_loader.py
def load_kraken_csv(file_path: str) -> pd.DataFrame:
    """Load Kraken CSV with validation."""
    df = pd.read_csv(file_path, sep='\t')  # Kraken uses tabs
    
    # Validate columns
    expected = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades']
    assert all(col in df.columns for col in expected)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    
    # Validate price consistency
    assert (df['high'] >= df['low']).all()
    assert (df['close'] >= df['low']).all()
    assert (df['close'] <= df['high']).all()
    
    return df.sort_values('timestamp').reset_index(drop=True)
```

### Feature Extraction

```python
# src/data/feature_extractor.py
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract scale-free features."""
    result = df.copy()
    
    # Returns
    result['return'] = np.log(result['close'] / result['close'].shift(1))
    
    # Range
    result['range'] = np.log(result['high'] / result['low'])
    
    # Body
    result['body'] = np.log(result['close'] / result['open'])
    
    # Volume change
    result['dlog_volume'] = np.log(result['volume']) - np.log(result['volume'].shift(1))
    
    # Rolling stats
    for window in [7, 30]:
        result[f'ret_mean_{window}'] = result['return'].rolling(window).mean()
        result[f'ret_std_{window}'] = result['return'].rolling(window).std()
    
    return result.dropna().reset_index(drop=True)
```

## Model Training Implementation

### Training Configuration

```python
# src/training/config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_type: str = "patchtst"  # "patchtst" | "nhits" | "tsmixer"
    context_length: int = 128
    horizon: int = 1
    hidden_size: int = 512
    num_layers: int = 6
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100
    device: str = "cuda"  # or "cpu"
    quantiles: list = None  # [0.1, 0.5, 0.9]
    
    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
```

### Training Pipeline

```python
# src/training/trainer.py
def train_model(config: TrainingConfig, train_data: pd.DataFrame, 
                val_data: pd.DataFrame) -> Tuple[Model, dict]:
    """Train model and return model + metadata."""
    
    # Create model
    model = create_model(config)
    
    # Prepare data for NeuralForecast
    # (convert to NeuralForecast format)
    
    # Train
    model.fit(train_data)
    
    # Evaluate on validation
    metrics = evaluate_model(model, val_data)
    
    # Save model
    save_model(model, config, metrics)
    
    return model, metrics
```

## Prediction Implementation

### Model Loading

```python
# src/prediction/model_loader.py
def load_model(model_path: str) -> Tuple[Model, dict]:
    """Load saved model and metadata."""
    # Load PyTorch model
    model = torch.load(f"{model_path}.pt")
    
    # Load metadata
    with open(f"{model_path}_metadata.json") as f:
        metadata = json.load(f)
    
    # Load scaler
    with open(f"{model_path}_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    return model, metadata, scaler
```

### Signal Generation

```python
# src/prediction/signal_generator.py
def generate_signal(prediction: dict, threshold: float = 0.005) -> str:
    """
    Convert prediction to signal.
    
    Args:
        prediction: {q10: float, q50: float, q90: float}
        threshold: Minimum return threshold
        
    Returns:
        "long" | "short" | "flat"
    """
    q10, q50, q90 = prediction['q10'], prediction['q50'], prediction['q90']
    
    # Long: median positive + limited downside
    if q50 > threshold and q10 > -threshold:
        return "long"
    
    # Short: median negative + limited upside
    if q50 < -threshold and q90 < threshold:
        return "short"
    
    # Flat: otherwise
    return "flat"
```

## Backtesting Implementation

### Walk-Forward Backtest

```python
# src/backtest/walk_forward.py
def walk_forward_backtest(model: Model, data: pd.DataFrame, 
                         train_window: int, test_window: int) -> dict:
    """Perform walk-forward backtest."""
    results = []
    
    start_idx = 0
    while start_idx + train_window + test_window <= len(data):
        # Split data
        train_data = data[start_idx:start_idx + train_window]
        test_data = data[start_idx + train_window:start_idx + train_window + test_window]
        
        # Train model
        model.fit(train_data)
        
        # Generate predictions for test period
        predictions = model.predict(test_data)
        
        # Execute signals
        trades = execute_signals(predictions, test_data)
        
        # Calculate metrics
        metrics = calculate_metrics(trades)
        results.append(metrics)
        
        # Roll forward
        start_idx += test_window
    
    # Aggregate results
    return aggregate_results(results)
```

## Terminal Interface Implementation

### Main Menu

```python
# src/cli/menu.py
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def show_main_menu() -> str:
    """Display main menu and return selection."""
    console.print(Panel.fit("[bold]Bitcoin Trading Model[/bold]", style="blue"))
    
    table = Table(show_header=False, box=None)
    table.add_row("1", "Train Model")
    table.add_row("2", "Make Prediction")
    table.add_row("3", "Run Backtest")
    table.add_row("4", "View Data")
    table.add_row("5", "Model Info")
    table.add_row("6", "Exit")
    
    console.print(table)
    
    choice = console.input("\n[bold]Select option:[/bold] ")
    return choice
```

## CLI Implementation

### Argument Parser

```python
# src/cli/cli.py
import click

@click.group()
def cli():
    """Bitcoin Trading Model CLI."""
    pass

@cli.command()
@click.option('--config', type=click.Path(exists=True))
def train(config):
    """Train a new model."""
    # Training logic
    pass

@cli.command()
@click.option('--model', required=True)
@click.option('--data', default='latest')
def predict(model, data):
    """Generate prediction."""
    # Prediction logic
    pass

@cli.command()
@click.option('--model', required=True)
@click.option('--start', required=True)
@click.option('--end', required=True)
def backtest(model, start, end):
    """Run backtest."""
    # Backtest logic
    pass

@cli.command()
@click.option('--port', default=8000)
def api(port):
    """Start API server."""
    # Start FastAPI server
    pass

if __name__ == '__main__':
    cli()
```

## API Implementation

### FastAPI App

```python
# src/api/main.py
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="Bitcoin Trading Model API",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Bitcoin Trading Model API"}
```

### API Routes

```python
# src/api/routes.py
from fastapi import APIRouter, HTTPException
from src.api.models import PredictionRequest, PredictionResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate prediction from latest data."""
    try:
        # Load model
        model, metadata, scaler = load_model(request.model_id)
        
        # Get latest data
        data = get_latest_data(metadata['context_length'])
        
        # Generate prediction
        prediction = generate_prediction(model, data, scaler)
        
        # Convert to signal
        signal = generate_signal(prediction, request.threshold)
        
        return PredictionResponse(
            prediction=prediction,
            signal=signal,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Strategy

### Test Structure

- **Unit Tests:** Test individual functions in isolation
- **Integration Tests:** Test component interactions
- **End-to-End Tests:** Test full workflows (train → predict → backtest)

### Test Coverage

- **Target:** >90% code coverage
- **Critical Paths:** 100% coverage (data pipeline, training, prediction)

### Example Test

```python
# tests/test_feature_extractor.py
def test_extract_features():
    """Test feature extraction."""
    df = create_sample_candles()
    result = extract_features(df)
    
    assert 'return' in result.columns
    assert 'range' in result.columns
    assert not result['return'].isna().any()
```

## Deployment Strategy

### Development

- Run locally with `python -m src.cli --menu`
- API server: `python -m src.cli --api --port 8000`

### Production

- **Database:** PostgreSQL on dedicated server
- **API:** FastAPI with uvicorn, behind nginx
- **Training:** Run on GPU server (RTX 3090)
- **Monitoring:** Log all operations, track model performance

### Model Retraining

- **Schedule:** Weekly or monthly retraining
- **Trigger:** When new data available or performance degrades
- **Process:** Automated training pipeline with validation

