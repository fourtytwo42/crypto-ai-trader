"""Pump.fun database connections."""

from __future__ import annotations

from src.config import get_settings
from src.database.connection import DatabaseManager

_pumpfun_manager: DatabaseManager | None = None


def get_pumpfun_db_manager() -> DatabaseManager:
    settings = get_settings()
    if not settings.pumpfun_database_url:
        raise RuntimeError(
            "pumpfun_database_url is not set. Add PUMPFUN_DATABASE_URL to your .env."
        )
    global _pumpfun_manager
    if _pumpfun_manager is None:
        _pumpfun_manager = DatabaseManager(settings.pumpfun_database_url)
    return _pumpfun_manager
