# Bitcoin Trading Model - Architecture

**Complete system design, component breakdown, data flow, technology choices, and design decisions.**

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Technology Choices](#technology-choices)
5. [Design Decisions](#design-decisions)
6. [Feature Engineering](#feature-engineering)
7. [Model Architecture](#model-architecture)
8. [Prediction Pipeline](#prediction-pipeline)
9. [Backtesting Architecture](#backtesting-architecture)
10. [Error Handling](#error-handling)

## System Overview

The system is organized into distinct layers:

1. **Data Layer** - PostgreSQL database storing raw and processed data
2. **Processing Layer** - Feature engineering, model training, prediction generation
3. **Interface Layer** - Terminal menu, CLI arguments, REST API
4. **Evaluation Layer** - Backtesting engine with walk-forward evaluation

## Component Architecture

### 1. Data Pipeline (`src/data/`)

**Purpose:** Load, validate, and store candle data from CSV files.

**Components:**
- `csv_loader.py` - Loads Kraken CSV format, validates schema
- `feature_extractor.py` - Computes log returns, range, volume features
- `normalizer.py` - Rolling normalization (z-score, robust scaling)
- `database.py` - Database connection and operations

**Key Functions:**
```python
def load_kraken_csv(file_path: str) -> pd.DataFrame
def extract_features(df: pd.DataFrame) -> pd.DataFrame
def normalize_features(df: pd.DataFrame, method: str = "rolling_zscore") -> pd.DataFrame
def save_to_database(df: pd.DataFrame, table: str) -> None
```

### 2. Model Training Pipeline (`src/training/`)

**Purpose:** Train time-series forecasting models with proper train/val/test splits.

**Components:**
- `trainer.py` - Main training orchestration
- `data_preparation.py` - Creates train/val/test splits with walk-forward logic
- `model_factory.py` - Creates PatchTST, NHITS, TSMixer models
- `evaluator.py` - Evaluates model performance on validation/test sets

**Key Functions:**
```python
def prepare_training_data(df: pd.DataFrame, train_end: str, val_end: str, test_end: str) -> Tuple
def train_model(model_type: str, config: dict, train_data: pd.DataFrame) -> Model
def evaluate_model(model: Model, test_data: pd.DataFrame) -> dict
def save_model(model: Model, scaler: object, metadata: dict, path: str) -> None
```

### 3. Prediction Engine (`src/prediction/`)

**Purpose:** Load trained models and generate predictions with uncertainty.

**Components:**
- `model_loader.py` - Loads saved models and metadata
- `predictor.py` - Generates forecasts (point estimates + quantiles)
- `signal_generator.py` - Converts predictions to directional signals (long/short/flat)

**Key Functions:**
```python
def load_model(model_path: str) -> Tuple[Model, dict]
def predict(model: Model, features: np.ndarray, horizon: int) -> dict
def generate_signal(prediction: dict, threshold: float) -> str  # "long" | "short" | "flat"
```

### 4. Backtest Engine (`src/backtest/`)

**Purpose:** Walk-forward evaluation with proper leakage prevention.

**Components:**
- `walk_forward.py` - Implements rolling window evaluation
- `signal_executor.py` - Simulates trading with fees/slippage
- `metrics_calculator.py` - Computes Sharpe, win rate, max drawdown, etc.

**Key Functions:**
```python
def walk_forward_backtest(model: Model, data: pd.DataFrame, train_window: int, test_window: int) -> dict
def execute_signals(signals: pd.DataFrame, prices: pd.DataFrame, fees: float, slippage: float) -> pd.DataFrame
def calculate_metrics(returns: pd.Series) -> dict
```

### 5. Terminal Interface (`src/cli/`)

**Purpose:** Interactive menu-driven interface.

**Components:**
- `menu.py` - Main menu loop with Rich terminal UI
- `train_menu.py` - Training configuration and execution
- `predict_menu.py` - Prediction interface
- `backtest_menu.py` - Backtesting configuration
- `data_menu.py` - Data viewing and management

**Key Functions:**
```python
def show_main_menu() -> str
def show_train_menu() -> None
def show_predict_menu() -> None
def show_backtest_menu() -> None
```

### 6. CLI Arguments (`src/cli/`)

**Purpose:** Non-interactive mode for automation.

**Components:**
- `cli.py` - Argument parser (Click or argparse)
- `commands.py` - Command implementations

**Commands:**
```bash
python -m src.cli --train --config config.json
python -m src.cli --predict --model model_v1.pt --data latest
python -m src.cli --backtest --model model_v1.pt --start 2020-01-01 --end 2023-12-31
python -m src.cli --api --port 8000
python -m src.cli --menu  # Interactive mode (default)
```

### 7. FastAPI Server (`src/api/`)

**Purpose:** REST API for programmatic access.

**Components:**
- `main.py` - FastAPI app initialization
- `routes.py` - API endpoint definitions
- `models.py` - Pydantic request/response models

**Endpoints:**
- `POST /predict` - Generate prediction from latest data
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /models/{model_id}` - Get model metadata

## Data Flow

### Training Flow

```
CSV File → Load & Validate → Extract Features → Normalize → Save to DB
                                                              ↓
                                    Train/Val/Test Split (Walk-Forward)
                                                              ↓
                                    Train Model (PatchTST/NHITS)
                                                              ↓
                                    Evaluate on Validation Set
                                                              ↓
                                    Save Model + Metadata
```

### Prediction Flow

```
Latest Data from DB → Extract Features → Normalize (using saved scaler)
                                                      ↓
                                    Load Model + Metadata
                                                      ↓
                                    Generate Prediction (quantiles)
                                                      ↓
                                    Convert to Signal (long/short/flat)
                                                      ↓
                                    Return Prediction + Signal
```

### Backtesting Flow

```
Historical Data from DB → Walk-Forward Windows
                                    ↓
                        Train on Window A → Test on Window B
                                    ↓
                        Generate Predictions for Test Window
                                    ↓
                        Execute Signals (with fees/slippage)
                                    ↓
                        Calculate Metrics (Sharpe, win rate, P&L)
                                    ↓
                        Roll Forward → Repeat
                                    ↓
                        Aggregate Results
```

## Technology Choices

### NeuralForecast

**Why:** Provides state-of-the-art time-series models (PatchTST, NHITS, TSMixer) with unified API, built-in probabilistic forecasting, and efficient training.

**Alternatives Considered:**
- Darts: Good but less modern models
- PyTorch Forecasting: More complex, TFT-focused
- Custom implementation: Too much work

**Decision:** Use NeuralForecast for v1, can switch models easily later.

### PostgreSQL

**Why:** 
- Relational structure for candles, features, predictions, backtest results
- Easy to query historical data for training
- Can add TimescaleDB extension later for time-series optimization
- Familiar and reliable

**Alternatives Considered:**
- SQLite: Too limited for production
- InfluxDB: Overkill for this use case
- Parquet files: Less queryable, harder to update

**Decision:** PostgreSQL for flexibility and queryability.

### FastAPI

**Why:**
- Modern Python async framework
- Automatic OpenAPI docs
- Type validation with Pydantic
- Easy to add authentication later

**Alternatives Considered:**
- Flask: Older, less type-safe
- Django REST: Too heavy for simple API

**Decision:** FastAPI for modern Python best practices.

### Rich (Terminal UI)

**Why:**
- Beautiful terminal output (tables, progress bars, colors)
- Cross-platform
- Easy to use

**Alternatives Considered:**
- Plain print statements: Too basic
- curses: Too complex

**Decision:** Rich for professional terminal interface.

## Design Decisions

### 1. Scale-Free Features

**Decision:** Use log returns, not raw prices.

**Rationale:**
- Model generalizes across price levels (works at $10k or $200k)
- Returns are more stationary than prices
- Standard practice in financial ML

**Implementation:**
- `r_t = log(C_t / C_{t-1})` for returns
- `range_t = log(H_t / L_t)` for range/volatility
- `body_t = log(C_t / O_t)` for body size

### 2. No-Trade Zone

**Decision:** Only trade when predicted return exceeds threshold τ.

**Rationale:**
- Avoids trading on tiny random moves
- Improves win rate by filtering low-confidence predictions
- Accounts for fees/slippage

**Implementation:**
- Long if `q50 > τ` AND `q10 > -τ` (upside bias with limited downside)
- Short if `q50 < -τ` AND `q90 < τ` (downside bias with limited upside)
- Flat otherwise

### 3. Walk-Forward Evaluation

**Decision:** Use rolling windows, not single train/test split.

**Rationale:**
- Markets drift over time
- Single split doesn't test robustness
- More realistic evaluation

**Implementation:**
- Train on 2013-2019, validate on 2020-2021, test on 2022-2023
- Then roll forward: train on 2014-2020, validate on 2021-2022, test on 2023
- Repeat for multiple folds

### 4. Purged/Embargo Splits

**Decision:** Use purging and embargo periods to prevent leakage.

**Rationale:**
- Overlapping horizons can leak future info
- Standard practice in financial ML

**Implementation:**
- Purge: Remove training samples that overlap with test labels
- Embargo: Add gap between train and test periods

### 5. Probabilistic Forecasting

**Decision:** Predict quantiles (0.1, 0.5, 0.9), not just point estimates.

**Rationale:**
- Provides uncertainty estimates
- Enables better risk management
- Helps with signal generation (use quantiles for confidence)

**Implementation:**
- NeuralForecast supports quantile loss
- Output: `{q10: -0.02, q50: 0.01, q90: 0.04}`

## Feature Engineering

### Input Features (Channels)

1. **Return:** `r_t = log(C_t / C_{t-1})`
2. **Range:** `range_t = log(H_t / L_t)`
3. **Body:** `body_t = log(C_t / O_t)`
4. **Volume Change:** `dlogV_t = log(V_t) - log(V_{t-1})`
5. **Rolling Stats (7-day):**
   - `ret_mean_7 = mean(r_{t-7..t-1})`
   - `ret_std_7 = std(r_{t-7..t-1})`
6. **Rolling Stats (30-day):**
   - `ret_mean_30 = mean(r_{t-30..t-1})`
   - `ret_std_30 = std(r_{t-30..t-1})`

### Normalization

**Method:** Rolling z-score (using only past data to prevent leakage)

```python
z_t = (x_t - mean(x_{t-N..t-1})) / std(x_{t-N..t-1})
```

**Window:** N = 30 days (or use train-set statistics for inference)

### Target

**Primary:** Forward log return `r_{t+1} = log(C_{t+1} / C_t)`

**Extended:** Can predict `r_{t+1..t+k}` for multi-horizon (k = 3, 7 days)

## Model Architecture

### PatchTST Configuration (Default)

- **Hidden Size:** 512
- **Layers:** 6-8
- **Patch Length:** 16 (for daily data, ~2 weeks per patch)
- **Stride:** 8
- **Context Length:** 128 days (~4 months)
- **Horizon:** 1 day (can extend to 3-7)
- **Parameters:** ~15-25M

### NHITS Configuration (Alternative)

- **Hidden Size:** 512
- **Layers:** 3-4 (hierarchical)
- **Context Length:** 128 days
- **Horizon:** 1 day
- **Parameters:** ~10-15M

### Training Configuration

- **Loss:** Quantile loss (0.1, 0.5, 0.9)
- **Optimizer:** AdamW
- **Learning Rate:** 1e-4 (with scheduler)
- **Batch Size:** 32-64
- **Epochs:** 50-100 (with early stopping)
- **Regularization:** Dropout 0.1, weight decay 1e-5

## Prediction Pipeline

### Step 1: Load Latest Data

```python
# Get last N days from database
data = db.get_latest_candles(n=context_length)
```

### Step 2: Extract Features

```python
features = extract_features(data)
```

### Step 3: Normalize

```python
# Use saved scaler from training
normalized = scaler.transform(features)
```

### Step 4: Generate Prediction

```python
prediction = model.predict(normalized, horizon=1)
# Returns: {q10: float, q50: float, q90: float}
```

### Step 5: Convert to Signal

```python
signal = generate_signal(prediction, threshold=0.005)
# Returns: "long" | "short" | "flat"
```

## Backtesting Architecture

### Walk-Forward Procedure

1. **Initial Split:**
   - Train: 2013-10-06 to 2019-12-31
   - Validate: 2020-01-01 to 2021-12-31
   - Test: 2022-01-01 to 2023-12-31

2. **Roll Forward:**
   - Train: 2014-01-01 to 2020-12-31
   - Validate: 2021-01-01 to 2022-12-31
   - Test: 2023-01-01 to 2023-12-31

3. **Repeat:** Continue rolling until data exhausted

### Signal Execution

**Assumptions:**
- Entry: Next candle open after signal
- Exit: Next candle open after opposite signal or end of horizon
- Fees: 0.1% per trade (configurable)
- Slippage: 0.05% (configurable)

**P&L Calculation:**
```python
pnl = position * (exit_price - entry_price) - fees - slippage
```

### Metrics

- **Win Rate:** % of profitable trades
- **Average Win/Loss:** Mean profit vs mean loss
- **Sharpe Ratio:** Risk-adjusted returns
- **Max Drawdown:** Largest peak-to-trough decline
- **Total Return:** Cumulative P&L
- **Turnover:** Number of trades

## Error Handling

### Data Validation

- **CSV Format:** Validate columns, data types, missing values
- **Price Consistency:** High >= Low, Close in [Low, High]
- **Timestamp Ordering:** Ensure chronological order

### Model Errors

- **Missing Model:** Graceful error if model file not found
- **Version Mismatch:** Check model metadata version
- **Insufficient Data:** Require minimum context length

### API Errors

- **400 Bad Request:** Invalid input parameters
- **404 Not Found:** Model not found
- **500 Internal Server Error:** Model prediction failure

### Database Errors

- **Connection Failures:** Retry with exponential backoff
- **Transaction Errors:** Rollback and log
- **Constraint Violations:** Prevent duplicate data insertion

