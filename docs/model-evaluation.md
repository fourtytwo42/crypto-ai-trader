## Model Evaluation Guide

This guide covers how to evaluate your trained pump.fun models using both holdout tokens (reserved during training) and reserve tokens (completely separate test set).

### Overview

When training models, we use two types of evaluation tokens:

1. **Holdout tokens** - Tokens reserved during training (last 12 tokens from the qualified set)
   - Used for validation during training
   - May have been indirectly influenced during model selection
   - Good for quick evaluation

2. **Reserve tokens** - Tokens that meet minimum data requirements but were **never used in any training**
   - Completely unbiased evaluation data
   - Best for final model performance assessment
   - Helps detect if models have overfit to holdout tokens

### Selecting Reserve Tokens

Before evaluating on reserve tokens, you need to select them:

```bash
python -m pumpfun_train.cli_main pumpfun-select-reserve --count 50
```

This command:
- Finds tokens that meet minimum data requirements (290+ rows)
- Excludes any tokens that were used in training or holdout sets
- Selects the top N tokens by data count
- Saves them to `pumpfun_train/reserve_tokens.txt`

**Options:**
```bash
# Select more tokens (default: 50)
python -m pumpfun_train.cli_main pumpfun-select-reserve --count 100

# Re-select if you want different reserve tokens
python -m pumpfun_train.cli_main pumpfun-select-reserve --count 50
```

### Testing Individual Models

#### Using Holdout Tokens (Default)

```bash
python -m pumpfun_train.cli_main pumpfun-backtest \
  --model-dir pumpfun_train/models/regression/h10 \
  --minutes 10 \
  --max-samples 5000
```

#### Using Reserve Tokens

```bash
python -m pumpfun_train.cli_main pumpfun-backtest \
  --model-dir pumpfun_train/models/regression/h10 \
  --minutes 10 \
  --use-reserve \
  --max-samples 5000
```

**Options:**
- `--model-dir` - Model directory path (required)
- `--minutes` - Prediction horizon in minutes (default: 10)
- `--test-window` - Number of recent rows to use for testing (default: 240)
- `--max-tokens` - Maximum number of tokens to test (default: all)
- `--max-samples` - Maximum number of prediction samples to generate (default: unlimited)
- `--use-reserve` - Use reserve tokens instead of holdout tokens

**Metrics Returned:**
- `mae` - Mean Absolute Error
- `rmse` - Root Mean Squared Error
- `smape` - Symmetric Mean Absolute Percentage Error (%)
- `direction_accuracy` - % of predictions that correctly predict price direction (up/down)
- `price_accuracy_pct` - % accuracy of predicted price vs actual price
- `samples` - Number of predictions tested

### Testing All 20 Models

Use the comprehensive test script to evaluate all regression models at once:

```bash
python3 pumpfun_train/test_all_models.py
```

This script will:
1. Prompt you to choose between holdout or reserve tokens
2. Test all 20 models (h01 through h20)
3. Display a comprehensive results table
4. Show average metrics across all models
5. Identify best and worst performing models

**Example Output:**
```
================================================================================
TESTING ALL 20 PRICE PREDICTION MODELS
================================================================================
Choose evaluation tokens:
  1. Holdout tokens (used during training validation, default)
  2. Reserve tokens (completely separate, never seen during training)

Enter choice (1 or 2, default=1): 2
Using reserve tokens (completely separate from training)...
✓ Found 50 reserve tokens
This may take several minutes...

Testing models: 100%|████████| 20/20 [45:23<00:00]
```

**Results Table:**
```
Horizon    Direction    Price Acc    MAE          RMSE         SMAPE        Samples    Status
(Testing on Reserve tokens - 50 tokens available)
--------------------------------------------------------------------------------
  1 min   52.34%       45.67%       0.029926     0.053121     12.45%        1,234     ✓
  2 min   51.89%       44.23%       0.030694     0.051820     13.12%        1,189     ✓
...
```

**Summary Metrics:**
- Average Direction Accuracy: % across all models
- Average Price Accuracy: % across all models
- Average MAE, RMSE, SMAPE
- Total Samples Tested
- Total Testing Time

**Best/Worst Models:**
- Identifies models with best/worst direction accuracy
- Identifies models with best/worst price accuracy

### Understanding Metrics

**Direction Accuracy:**
- Percentage of predictions that correctly predict price direction (up/down)
- Random would be ~50%
- Higher is better
- Useful for trading strategies based on direction

**Price Accuracy %:**
- Percentage accuracy of predicted price vs actual price
- Formula: `max(0, 1 - abs(predicted - actual) / actual) * 100`
- 100% = perfect prediction, 0% = very wrong
- Useful for assessing prediction quality

**MAE (Mean Absolute Error):**
- Average absolute difference between predicted and actual returns
- Lower is better
- Gives equal weight to all errors

**RMSE (Root Mean Squared Error):**
- Square root of average squared errors
- Penalizes large errors more than small ones
- Lower is better

**SMAPE (Symmetric MAPE):**
- Symmetric mean absolute percentage error
- Percentage-based error metric
- Lower is better
- Handles zero values better than MAPE

### Testing Classifier

The direction classifier can also be backtested:

```bash
python -m pumpfun_train.cli_main pumpfun-classify-backtest \
  --model-dir pumpfun_train/models/classifier \
  --max-samples 10000
```

**Metrics:**
- `directional_accuracy` - % of correct direction predictions
- `samples` - Number of predictions tested

### Tips for Evaluation

1. **Always test on reserve tokens for final evaluation**
   - Reserve tokens provide unbiased performance estimates
   - If performance drops significantly on reserve vs holdout, models may have overfit

2. **Compare metrics across horizons**
   - Shorter horizons (1-5 min) may have better direction accuracy
   - Longer horizons (15-20 min) may have higher prediction errors
   - Use metrics to choose best horizons for your use case

3. **Use sufficient samples**
   - More samples = more reliable metrics
   - Default 5000 samples is good for quick testing
   - Use more (10,000+) for final evaluation

4. **Check direction accuracy vs price accuracy**
   - High direction accuracy but low price accuracy suggests model predicts direction well but magnitude poorly
   - High price accuracy suggests good overall predictions

5. **Monitor both holdout and reserve performance**
   - Large gap between holdout and reserve metrics suggests overfitting
   - Similar metrics suggest good generalization

### Automation

You can automate evaluation in your training scripts:

```bash
# After training, test on reserve tokens
python -m pumpfun_train.cli_main pumpfun-select-reserve --count 50
python3 pumpfun_train/test_all_models.py <<< "2"  # Use reserve tokens
```

### Troubleshooting

**No reserve tokens found:**
- Run `pumpfun-select-reserve` first
- Check that models have been trained (holdout files exist)
- Verify database has enough tokens that weren't used in training

**Low sample count:**
- Increase `--test-window` to use more recent data per token
- Increase `--max-tokens` to test more tokens
- Remove `--max-samples` limit

**Memory issues:**
- Reduce `--max-samples` to limit memory usage
- Reduce `--max-tokens` to test fewer tokens at once
- Test models individually instead of all at once

