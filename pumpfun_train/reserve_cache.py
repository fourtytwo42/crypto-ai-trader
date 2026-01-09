"""Cache for reserve token evaluation data.

Caches prepared reserve token data to avoid reloading from database
for each model during backtesting.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pumpfun_train.data import PumpfunTrainingData

CACHE_DIR = Path(__file__).parent / ".training_cache"
CACHE_DIR.mkdir(exist_ok=True)

RESERVE_CACHE_FILE = CACHE_DIR / "reserve_tokens_data.pkl"


def load_reserve_cache() -> "PumpfunTrainingData | None":
    """Load cached reserve token data."""
    if not RESERVE_CACHE_FILE.exists():
        return None
    
    try:
        with open(RESERVE_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_reserve_cache(training_data: "PumpfunTrainingData") -> None:
    """Save reserve token data to cache."""
    RESERVE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESERVE_CACHE_FILE, "wb") as f:
        pickle.dump(training_data, f)


def invalidate_reserve_cache() -> None:
    """Invalidate reserve token cache."""
    if RESERVE_CACHE_FILE.exists():
        RESERVE_CACHE_FILE.unlink()

