#!/usr/bin/env python3
"""Script to sync database and train all pumpfun models.

Trains:
1. Direction classifier (up/down prediction)
2. Price regression models for horizons 1-20 minutes

Models are saved in pumpfun_train/models/ and can be copied to pumpfun_api/models/

Usage:
    python pumpfun_train/train_all_models.py

Or use the bash script:
    bash pumpfun_train/train_all.sh
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable=None, **kwargs):
        if iterable is None:
            return DummyProgressBar()
        return iterable
    
    class DummyProgressBar:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_description(self, desc):
            pass
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass


# Removed unused function


def main() -> None:
    """Main training workflow."""
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Step 1: Sync database with new trades
    print("\n" + "="*60)
    print("STEP 1: Syncing database with new trades")
    print("="*60)
    start_time = time.time()
    from pumpfun_train.cli.commands import pumpfun_sync_command
    result = pumpfun_sync_command()
    elapsed = time.time() - start_time
    print(f"\n✓ Sync complete in {elapsed:.1f}s: tokens={result['tokens']} candles={result['candles']} features={result['features']}")
    
    # Step 2: Train direction classifier
    print("\n" + "="*60)
    print("STEP 2: Training direction classifier")
    print("="*60)
    classifier_dir = models_dir / "classifier"
    from pumpfun_train.cli.commands import pumpfun_classify_train_command
    start_time = time.time()
    result = pumpfun_classify_train_command(
        model_dir=str(classifier_dir),
        horizon_minutes=10,
        max_samples=10_000_000,  # Limit to 10M samples to prevent OOM
    )
    elapsed = time.time() - start_time
    print(f"\n✓ Classifier training complete in {elapsed:.1f}s")
    print(f"  Metrics: {result['metrics']}")
    print(f"  Holdout tokens: {len(result['holdout_tokens'])}")
    
    # Step 3: Train regression models for horizons 1-20
    print("\n" + "="*60)
    print("STEP 3: Training regression models for horizons 1-20 minutes")
    print("="*60)
    regression_dir = models_dir / "regression"
    regression_dir.mkdir(exist_ok=True)
    
    from pumpfun_train.cli.commands import pumpfun_train_command
    
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
        result = pumpfun_train_command(
            model_dir=str(horizon_dir),
            horizon_minutes=horizon,
            context_length=336,  # Best directional + price config (production NHITS)
            model_type="nhits",
            target_mode="sum",
            hidden_size=512,
            num_layers=3,
            epochs=50,
            batch_size=16,
            learning_rate=5e-5,
            holdout_count=12,
            nhits_stack_types=["identity", "identity", "identity"],
            nhits_n_blocks=[3, 2, 2],
            nhits_mlp_units=[[768, 768], [768, 768], [768, 768]],
            nhits_n_pool_kernel_size=[2, 2, 1],
            nhits_n_freq_downsample=[4, 2, 1],
        )
        elapsed = time.time() - start_time
        mae = result['metrics'].get('mae', 0)
        model_pbar.set_postfix({"MAE": f"{mae:.6f}", "time": f"{elapsed:.0f}s"})
    
    model_pbar.close()
    print(f"\n✓ All 20 regression models complete")
    
    print(f"\n{'='*60}")
    print("All models trained successfully!")
    print(f"{'='*60}")
    print(f"\nModels saved to: {models_dir}")
    print(f"  - Classifier: {classifier_dir}")
    print(f"  - Regression models: {regression_dir}/h01 through h20")
    # Step 4: Copy models to API location
    print(f"\n{'='*60}")
    print("STEP 4: Copying models to API location")
    print(f"{'='*60}")
    
    api_regression_dir = project_root / "pumpfun_api" / "models" / "regression"
    api_classifier_dir = project_root / "pumpfun_api" / "models" / "classifier"
    
    api_regression_dir.mkdir(parents=True, exist_ok=True)
    api_classifier_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    # Copy regression models
    print("Copying regression models to API...")
    regression_dirs = [d for d in regression_dir.iterdir() if d.is_dir() and d.name.startswith("h")]
    copy_pbar = tqdm(regression_dirs, desc="Copying models", unit="model", ncols=100)
    for horizon_dir in copy_pbar:
        copy_pbar.set_description(f"Copying {horizon_dir.name}")
        dest = api_regression_dir / horizon_dir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(horizon_dir, dest)
    copy_pbar.close()
    print("✓ Regression models copied")
    
    # Copy classifier
    print("Copying classifier to API...")
    if classifier_dir.exists():
        items = list(classifier_dir.iterdir())
        classifier_pbar = tqdm(items, desc="Copying classifier", unit="file", ncols=100)
        for item in classifier_pbar:
            classifier_pbar.set_description(f"Copying {item.name}")
            if item.is_file():
                shutil.copy2(item, api_classifier_dir / item.name)
            elif item.is_dir():
                dest = api_classifier_dir / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
        classifier_pbar.close()
    print("✓ Classifier copied")
    
    print("\nAll done! Models are ready for the API.")


if __name__ == "__main__":
    main()
