# Training All Pumpfun Models

This guide shows how to sync the database and train all models for the pumpfun API.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variable:
```bash
export PUMPFUN_DATABASE_URL="postgresql://user:password@localhost:5432/pumpfun_db"
# Or add to .env file in pumpfun_train/ directory
```

## Database Management

### Clearing Processed Data

If you need to start fresh or suspect duplicate data in the processed tables, you can clear them safely:

```bash
python -m pumpfun_train.cli.cli pumpfun-clear-processed --force
```

**Important:** This only clears derived data (`pump_candles_1m` and `pump_features_1m`), NOT the original pump.fun trade data (`trades` and `tokens` tables). Safe to run anytime.

Uses `TRUNCATE` for instant deletion - much faster than counting and deleting individual rows.

### Incremental Sync

The sync process is now **incremental** by default:
- Only processes tokens with new trades since the last sync
- Skips tokens that already have complete candle/feature data
- Checks for duplicates before inserting
- Subsequent runs are much faster (only processes new data)

## Step 1: Sync Database

Sync new trades into candles/features with parallel processing:

```bash
python -m pumpfun_train.cli.cli pumpfun-sync
```

**Features:**
- **Parallel Processing:** Uses multiple CPU cores (default: CPU count, max 8 workers)
- **Progress Bars:** Shows token processing progress with ETA and rates
- **Incremental:** Only processes new data since last sync (no duplicates)
- **Smart Skipping:** Automatically skips tokens that are already up-to-date

**Options:**
```bash
# Limit tokens for testing
python -m pumpfun_train.cli.cli pumpfun-sync --max-tokens 100

# Control parallel workers
python -m pumpfun_train.cli.cli pumpfun-sync --max-workers 4

# Force full reprocess (replaces existing data)
python -m pumpfun_train.cli.cli pumpfun-sync --replace

# Skip SOL/USD price lookup (faster but may have missing USD values)
python -m pumpfun_train.cli.cli pumpfun-sync --skip-price-lookup
```

**Output Example:**
```
Syncing tokens (8 workers): 100%|██████████| 26559/26559 [11:08<00:00, 39.73token/s]
Pump.fun sync complete: tokens=21987 candles=14103892 features=13409995
```

## Step 2: Train Direction Classifier

Train the up/down direction classifier with progress indicators:

```bash
python -m pumpfun_train.cli.cli pumpfun-classify-train \
  --model-dir pumpfun_train/models/classifier \
  --horizon-minutes 10
```

**Features:**
- Progress bar for training epochs
- Per-epoch batch progress showing current loss
- Threshold search progress during validation
- Data preparation status messages

**Output Example:**
```
Preparing training data from 21987 tokens...
Training data prepared: 13409995 samples, 42 features
Training epochs: 100%|██████████| 20/20 [05:23<00:00, 16.15s/epoch]
Epoch 1/20: 100%|██████████| 26191/26191 [00:32<00:00, loss=0.123456]
Validating and finding optimal threshold...
Searching threshold: 100%|██████████| 33/33 [00:02<00:00]
```

## Step 3: Train Regression Models (Horizons 1-20)

Train regression models for each horizon (1-20 minutes). The API expects models in `pumpfun_api/models/regression/h{01-20}/` format.

```bash
# Train all 20 models manually
for horizon in {1..20}; do
  printf -v h "%02d" $horizon
  python -m pumpfun_train.cli.cli pumpfun-train \
    --model-dir pumpfun_train/models/regression/h$h \
    --horizon-minutes $horizon \
    --context-length 240 \
    --model-type nhits \
    --target-mode sum \
    --hidden-size 256 \
    --num-layers 2 \
    --epochs 30 \
    --batch-size 32 \
    --learning-rate 1e-4
done
```

## Step 4: Copy Models to API

After training, copy models to where the API expects them:
```bash
# Copy regression models
cp -r pumpfun_train/models/regression/* pumpfun_api/models/regression/

# Copy classifier
cp -r pumpfun_train/models/classifier/* pumpfun_api/models/classifier/
```

**Note:** The automated training script handles this automatically.

## Automated Training: One Command

Use the automated script which handles everything including progress bars and model copying:

```bash
bash pumpfun_train/train_all.sh
```

Or the Python version:

```bash
python pumpfun_train/train_all_models.py
```

**This will:**
1. **Sync database** with new trades (parallel processing, progress bars with ETA)
2. **Train classifier** (with epoch/batch progress, threshold search progress)
3. **Train all 20 regression models** (with progress bar for model training, shows MAE and time per model)
4. **Copy models to API** automatically (with progress bar for file copying)

**Full Output Example:**
```
============================================================
STEP 1: Syncing database with new trades
============================================================
Syncing tokens (8 workers): 100%|██████████| 26559/26559 [11:08<00:00, 39.73token/s]
Pump.fun sync complete: tokens=21987 candles=14103892 features=13409995

✓ Database sync complete

============================================================
STEP 2: Training direction classifier
============================================================
Preparing training data from 21987 tokens...
Training epochs: 100%|██████████| 20/20 [05:23<00:00]
✓ Classifier training complete

============================================================
STEP 3: Training regression models for horizons 1-20 minutes
============================================================
Training models: 100%|██████████| 20/20 [2:15:30<00:00, MAE=0.000123, time=406s]
✓ All 20 regression models complete

============================================================
STEP 4: Copying models to API location
============================================================
Copying regression models: 100%|██████████| 20/20 [00:05<00:00]
Copying classifier: 100%|██████████| 3/3 [00:01<00:00]
✓ All models copied

All done! Models are ready for the API.
```

## CLI Commands Reference

### Database Commands

- `pumpfun-sync` - Sync trades into candles/features (parallel, incremental, with progress)
- `pumpfun-clear-processed` - Clear processed tables (keeps original trades/tokens)

### Training Commands

- `pumpfun-classify-train` - Train direction classifier (up/down prediction)
- `pumpfun-train` - Train regression model for price prediction

### Evaluation Commands

- `pumpfun-select-reserve [--count 50]` - Select reserve tokens for evaluation (tokens never used in training)
- `pumpfun-backtest` - Backtest regression model on holdout or reserve tokens
  - `--use-reserve` - Use reserve tokens instead of holdout tokens (default: holdout)
- `pumpfun-classify-backtest` - Backtest direction classifier

### Prediction Commands

- `pumpfun-predict` - Predict price movement for a token

See `python -m pumpfun_train.cli.cli --help` for full command reference.

## Model Evaluation

After training, evaluate your models to assess performance. See the comprehensive [Model Evaluation Guide](../docs/model-evaluation.md) for detailed instructions.

### Quick Start

1. **Select reserve tokens** (completely separate from training):
   ```bash
   python -m pumpfun_train.cli_main pumpfun-select-reserve --count 50
   ```

2. **Test individual model**:
   ```bash
   # On holdout tokens (default)
   python -m pumpfun_train.cli_main pumpfun-backtest \
     --model-dir pumpfun_train/models/regression/h10 \
     --minutes 10
   
   # On reserve tokens (unbiased)
   python -m pumpfun_train.cli_main pumpfun-backtest \
     --model-dir pumpfun_train/models/regression/h10 \
     --minutes 10 \
     --use-reserve
   ```

3. **Test all 20 models**:
   ```bash
   python3 pumpfun_train/test_all_models.py
   ```

This comprehensive test script evaluates all regression models and provides detailed metrics including direction accuracy, price accuracy, MAE, RMSE, and SMAPE.

**See [docs/model-evaluation.md](../docs/model-evaluation.md) for complete evaluation guide.**

