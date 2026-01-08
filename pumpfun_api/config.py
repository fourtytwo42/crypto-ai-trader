from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

_ENV_PATH = Path(__file__).resolve().parent / ".env"
if load_dotenv is not None and _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


def get_pumpfun_database_url() -> str:
    url = os.getenv("PUMPFUN_DATABASE_URL")
    if not url:
        raise RuntimeError("PUMPFUN_DATABASE_URL is not set")
    return url


def get_regression_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "regression"


def get_model_dir() -> Path:
    # Backwards-compatible default model directory (regression root).
    return get_regression_models_dir()
