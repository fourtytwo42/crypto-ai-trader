#!/usr/bin/env python3
"""Script to train regression models for horizons 1-20 (Step 3 only).

Use this when classifier training is already complete and you only need to train
the regression models.
"""

from __future__ import annotations

import sys
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
from io import StringIO

# Completely suppress all logging output during training
# Redirect logging to a null handler
logging.basicConfig(
    level=logging.CRITICAL,  # Only show critical errors
    handlers=[logging.NullHandler()],
    force=True,  # Override any existing configuration
)
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger().handlers = [logging.NullHandler()]

# Suppress specific noisy loggers completely
for logger_name in [
    "pumpfun_train",
    "pumpfun_train.normalizer",
    "pumpfun_train.progress",
    "pumpfun_train.trainer",
    "pumpfun_train.data",
    "pumpfun_train.pipeline",
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False

# Suppress structlog by setting environment variable
os.environ["LOG_LEVEL"] = "CRITICAL"
os.environ["STRUCTLOG_LEVEL"] = "CRITICAL"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    """Train regression models for horizons 1-20."""
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    regression_dir = models_dir / "regression"
    regression_dir.mkdir(parents=True, exist_ok=True)
    
    from pumpfun_train.cli.commands import pumpfun_train_command
    
    print("="*60)
    print("Training regression models for horizons 1-20 minutes")
    print("="*60)
    print("This will train 20 models (one for each horizon)")
    print()
    
    # Progress bar for all regression models
    horizons = list(range(1, 21))
    model_pbar = tqdm(
        horizons,
        desc="Training models",
        unit="model",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}"
    )
    
    for horizon in model_pbar:
        horizon_dir = regression_dir / f"h{horizon:02d}"
        model_pbar.set_description(f"Training h{horizon:02d} ({horizon}min)")
        start_time = time.time()
        try:
            result = pumpfun_train_command(
                model_dir=str(horizon_dir),
                horizon_minutes=horizon,
                context_length=240,
                model_type="nhits",
                target_mode="sum",
                hidden_size=256,
                num_layers=2,
                epochs=30,
                batch_size=32,
                learning_rate=1e-4,
                holdout_count=12,
            )
            elapsed = time.time() - start_time
            mae = result['metrics'].get('mae', 0)
            model_pbar.set_postfix({"MAE": f"{mae:.6f}", "time": f"{elapsed:.0f}s"})
        except Exception as e:
            model_pbar.set_postfix({"status": "ERROR"})
            print(f"\n❌ Error training h{horizon:02d}: {e}")
            print(f"Continuing with next model...\n")
            continue
    
    model_pbar.close()
    print(f"\n✓ All 20 regression models complete")
    
    # Step 4: Copy models to API location
    print(f"\n{'='*60}")
    print("STEP 4: Copying models to API location")
    print(f"{'='*60}")
    
    project_root = Path(__file__).parent.parent
    api_regression_dir = project_root / "pumpfun_api" / "models" / "regression"
    api_classifier_dir = project_root / "pumpfun_api" / "models" / "classifier"
    
    api_regression_dir.mkdir(parents=True, exist_ok=True)
    api_classifier_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    # Copy regression models
    print("Copying regression models to API...")
    for horizon_dir in tqdm(list(regression_dir.iterdir()), desc="Copying regression models", unit="model", ncols=100):
        if horizon_dir.is_dir() and horizon_dir.name.startswith("h"):
            dest = api_regression_dir / horizon_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(horizon_dir, dest)
    print("✓ Regression models copied")
    
    # Copy classifier (if it exists)
    classifier_dir = models_dir / "classifier"
    if classifier_dir.exists():
        print("Copying classifier to API...")
        for item in tqdm(list(classifier_dir.iterdir()), desc="Copying classifier", unit="file", ncols=100):
            if item.is_file():
                shutil.copy2(item, api_classifier_dir / item.name)
            elif item.is_dir():
                dest = api_classifier_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
        print("✓ Classifier copied")
    
    print("\nAll done! Models are ready for the API.")


if __name__ == "__main__":
    main()

