## Pump.fun Training Workflow

This branch focuses on pump.fun minute-level models. The full training guide lives in `pumpfun_train/TRAINING_GUIDE.md`; the steps below summarize the flow.

### 1) Configure environment

```bash
export PUMPFUN_DATABASE_URL="postgresql://user:password@localhost:5432/pumpfun_db"
```

### 2) Database Management

#### Initial Setup: Clear Processed Data (Optional)

If you need to start fresh or suspect duplicate data, clear the processed tables (keeps original trades/tokens):

```bash
python -m pumpfun_train.cli.cli pumpfun-clear-processed --force
```

**Note:** This only clears derived data (candles/features), NOT the original pump.fun trade data. Safe to run anytime.

#### Sync trades into Postgres

Sync trades into candles/features with parallel processing and progress bars:

```bash
python -m pumpfun_train.cli.cli pumpfun-sync
```

**Features:**
- Parallel processing (uses multiple CPU cores, max 8 workers)
- Progress bars with ETA for token processing
- Incremental sync - only processes new trades since last sync (no duplicates)
- Automatically skips tokens that are already up-to-date

**Options:**
```bash
# Limit number of tokens to process (for testing)
python -m pumpfun_train.cli.cli pumpfun-sync --max-tokens 100

# Use fewer workers (default is CPU count, max 8)
python -m pumpfun_train.cli.cli pumpfun-sync --max-workers 4

# Replace existing data (forces full reprocess)
python -m pumpfun_train.cli.cli pumpfun-sync --replace
```

### 3) Train the direction classifier

Train the up/down direction classifier with progress indicators:

```bash
python -m pumpfun_train.cli.cli pumpfun-classify-train \
  --model-dir pumpfun_train/models/classifier \
  --horizon-minutes 10
```

**Features:**
- Progress bar for epochs
- Per-epoch batch progress with loss values
- Threshold search progress during validation
- Shows data preparation status

### 4) Train regression models (1-20 minute horizons)

Train regression models for each horizon (1-20 minutes):

```bash
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

### 5) Copy models to the API

After training, copy models to where the API expects them:

```bash
# Copy regression models
cp -r pumpfun_train/models/regression/* pumpfun_api/models/regression/

# Copy classifier
cp -r pumpfun_train/models/classifier/* pumpfun_api/models/classifier/
```

**Note:** The automated training script handles this automatically.

### Automated Training: One Command

Use the automated script which handles everything including copying models:

```bash
bash pumpfun_train/train_all.sh
```

Or the Python version:

```bash
python pumpfun_train/train_all_models.py
```

**This will:**
1. Sync database with new trades (with progress bars, parallel processing)
2. Train direction classifier (with epoch/batch progress)
3. Train all 20 regression models (with progress bar for models)
4. Copy models to API location automatically

**Progress Indicators:**
- Database sync shows token processing progress with ETA
- Classifier training shows epoch progress, batch progress, and threshold search
- Regression training shows overall model progress and per-model metrics

### 6) Evaluate Models

After training, evaluate your models on both holdout and reserve tokens:

#### Select Reserve Tokens

Reserve tokens are tokens that were never used in training, providing unbiased evaluation:

```bash
python -m pumpfun_train.cli_main pumpfun-select-reserve --count 50
```

#### Test Individual Models

```bash
# Test on holdout tokens (default)
python -m pumpfun_train.cli_main pumpfun-backtest \
  --model-dir pumpfun_train/models/regression/h10 \
  --minutes 10

# Test on reserve tokens (completely separate)
python -m pumpfun_train.cli_main pumpfun-backtest \
  --model-dir pumpfun_train/models/regression/h10 \
  --minutes 10 \
  --use-reserve
```

#### Test All Models

Test all 20 regression models at once with comprehensive metrics:

```bash
python3 pumpfun_train/test_all_models.py
```

This will prompt you to choose between:
- Holdout tokens (used during training validation)
- Reserve tokens (completely separate, never seen during training)

**See [Model Evaluation Guide](model-evaluation.md) for detailed evaluation instructions and metric explanations.**
