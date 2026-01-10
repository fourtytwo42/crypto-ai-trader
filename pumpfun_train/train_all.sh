#!/bin/bash
# Script to sync database and train all pumpfun models
# 
# Trains:
# 1. Direction classifier (up/down prediction)
# 2. Price regression models for horizons 1-20 minutes
#
# Models are saved in pumpfun_train/models/ and can be copied to pumpfun_api/models/

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
REGRESSION_DIR="$MODELS_DIR/regression"
CLASSIFIER_DIR="$MODELS_DIR/classifier"

mkdir -p "$MODELS_DIR"
mkdir -p "$REGRESSION_DIR"
mkdir -p "$CLASSIFIER_DIR"

echo "============================================================"
echo "STEP 1: Syncing database with new trades"
echo "============================================================"
echo "This will show progress bars for each token being processed..."
python -m pumpfun_train.cli_main pumpfun-sync
echo ""
echo "✓ Database sync complete"

echo ""
echo "============================================================"
echo "STEP 2: Training direction classifier"
echo "============================================================"
echo "Training classifier with progress indicators..."
python -m pumpfun_train.cli_main pumpfun-classify-train \
  --model-dir "$CLASSIFIER_DIR" \
  --horizon-minutes 10
echo ""
echo "✓ Classifier training complete"

echo ""
echo "============================================================"
echo "STEP 3: Training regression models for horizons 1-20 minutes"
echo "============================================================"

echo "Training 20 regression models (this will take a while)..."
echo "Each model will show training progress with epochs and metrics"
echo ""

for horizon in {1..20}; do
  printf -v h "%02d" $horizon
  horizon_dir="$REGRESSION_DIR/h$h"
  
  echo "[$horizon/20] Training ${horizon}-minute horizon model (h$h)..."
  python -m pumpfun_train.cli_main pumpfun-train \
    --model-dir "$horizon_dir" \
    --horizon-minutes $horizon \
    --context-length 336 \
    --model-type nhits \
    --target-mode sum \
    --hidden-size 512 \
    --num-layers 3 \
    --epochs 50 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --holdout-count 12
  
  echo "✓ [$horizon/20] ${horizon}-minute model complete"
done

echo ""
echo "============================================================"
echo "All models trained successfully!"
echo "============================================================"
echo ""
echo "Models saved to: $MODELS_DIR"
echo "  - Classifier: $CLASSIFIER_DIR"
echo "  - Regression models: $REGRESSION_DIR/h01 through h20"
echo ""
echo "============================================================"
echo "STEP 4: Copying models to API location"
echo "============================================================"

API_REGRESSION_DIR="$(cd "$SCRIPT_DIR/../pumpfun_api/models/regression" && pwd)"
API_CLASSIFIER_DIR="$(cd "$SCRIPT_DIR/../pumpfun_api/models/classifier" && pwd)"

mkdir -p "$API_REGRESSION_DIR"
mkdir -p "$API_CLASSIFIER_DIR"

# Copy regression models
echo "Copying regression models to API..."
cp -r "$REGRESSION_DIR"/* "$API_REGRESSION_DIR/" 2>/dev/null || true
echo "✓ Regression models copied"

# Copy classifier
echo "Copying classifier to API..."
cp -r "$CLASSIFIER_DIR"/* "$API_CLASSIFIER_DIR/" 2>/dev/null || true
echo "✓ Classifier copied"

echo ""
echo "All done! Models are ready for the API."
