# Database Management

Guide for managing pump.fun database tables and data synchronization.

## Table Structure

### Original Data Tables (DO NOT DELETE)

- `tokens` - Token metadata (mint address, symbol, name, creation timestamp)
- `trades` - Raw trade stream data (price, amount, timestamp per trade)

These tables contain the original pump.fun data and should never be deleted.

### Processed Data Tables (Safe to Clear)

- `pump_candles_1m` - Derived 1-minute candles (OHLCV data)
- `pump_features_1m` - Derived 1-minute features (technical indicators)
- `pump_sol_prices` - Cached SOL/USD hourly prices for normalization

These tables are generated from the original trade data and can be safely cleared and regenerated.

## Clearing Processed Data

If you need to start fresh or suspect duplicate data in the processed tables:

```bash
python -m pumpfun_train.cli.cli pumpfun-clear-processed --force
```

**What it does:**
- Uses `TRUNCATE` for instant deletion (much faster than DELETE)
- Clears `pump_candles_1m` table
- Clears `pump_features_1m` table
- **Does NOT** touch `trades` or `tokens` tables (original data preserved)

**After clearing:**
Run sync to regenerate from trades:
```bash
python -m pumpfun_train.cli.cli pumpfun-sync
```

## Incremental Sync

The sync process is **incremental** by default to avoid duplicates and speed up subsequent runs.

### How It Works

1. **Checks existing data:** For each token, checks the latest candle timestamp
2. **Only processes new trades:** Loads only trades after the latest candle (with small buffer for overlap)
3. **Prevents duplicates:** Checks for existing records before inserting
4. **Skips up-to-date tokens:** Tokens with no new trades since last sync are skipped entirely

### Benefits

- **No duplicates:** Duplicate checking prevents inserting the same candle/feature twice
- **Faster subsequent runs:** Only processes new data, not all data again
- **Automatic:** Works transparently - no configuration needed
- **Progress tracking:** Shows which tokens are being processed vs skipped

### Example

**First run:**
```
Syncing tokens (8 workers): 100%|██████████| 26559/26559 [11:08<00:00]
Processed: 21987 tokens, 14103892 candles, 13409995 features
```

**Second run (10 minutes later):**
```
Syncing tokens (8 workers): 100%|██████████| 123/123 [00:45<00:00]
Processed: 89 tokens, 4567 candles, 4321 features
```

Only processes tokens with new trades, much faster!

### Force Full Reprocess

If you want to reprocess all tokens regardless of existing data:

```bash
python -m pumpfun_train.cli.cli pumpfun-sync --replace
```

This will:
- Delete existing candles/features for each token
- Reprocess all trades from scratch
- Much slower but ensures complete reprocessing

## Parallel Processing

The sync process uses parallel processing to speed up token processing.

### Configuration

- **Default:** Uses all CPU cores (max 8 workers)
- **Automatic:** Caps at 8 workers to avoid database overload
- **Adjustable:** Use `--max-workers` to control parallelism

```bash
# Use 4 workers instead of default
python -m pumpfun_train.cli.cli pumpfun-sync --max-workers 4

# Use 1 worker (no parallelism, slower but less DB load)
python -m pumpfun_train.cli.cli pumpfun-sync --max-workers 1
```

### Why Max 8 Workers?

- Prevents database connection pool exhaustion
- Balances speed with database load
- Each worker maintains its own database connection
- More workers = faster but more DB connections needed

## Progress Indicators

All sync operations show progress with:
- Progress bars with percentage
- ETA (estimated time remaining)
- Processing rate (tokens/second)
- Current counts (processed tokens, candles, features)

Example output:
```
Syncing tokens (8 workers): 45%|████▌     | 11951/26559 [05:12<12:45, 19.23token/s]
Processed: 10234, Candles: 7234567, Features: 6789012, Workers: 8
```

## Troubleshooting

### Sync is Very Slow

- Check if incremental sync is working (should skip most tokens on second run)
- Reduce `--max-workers` if database is overloaded
- Check database connection pool size
- Consider using `--skip-price-lookup` if SOL/USD lookup is slow

### Duplicate Data Suspected

- Clear processed tables: `pumpfun-clear-processed --force`
- Run sync again: `pumpfun-sync`
- The duplicate checking should prevent new duplicates

### Memory Issues

- Reduce `--max-workers` (fewer parallel processes = less memory)
- Process in smaller batches with `--max-tokens`
- Check database connection pool settings

### Database Connection Errors

- Verify `PUMPFUN_DATABASE_URL` is set correctly
- Check database server is running and accessible
- Verify database user has proper permissions
- Check connection pool size in database configuration

