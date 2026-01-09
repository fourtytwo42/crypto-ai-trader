"""Minimal config for pumpfun_train database connection."""

from __future__ import annotations

import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
    from pathlib import Path
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except Exception:
    pass


@lru_cache
def get_pumpfun_database_url() -> str:
    """Get pumpfun database URL from environment."""
    url = os.getenv("PUMPFUN_DATABASE_URL")
    if not url:
        raise RuntimeError(
            "PUMPFUN_DATABASE_URL is not set. Add PUMPFUN_DATABASE_URL to your .env."
        )
    return url


@lru_cache
def get_train_device() -> str:
    """Get training device (cuda or cpu)."""
    return os.getenv("TRAIN_DEVICE", "cpu")

