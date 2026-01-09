"""Pump.fun database connections for training."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from pumpfun_train.config_db import get_pumpfun_database_url


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str):
        """Initialize with database URL."""
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional scope around a series of operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


_pumpfun_manager: DatabaseManager | None = None


def get_pumpfun_db_manager() -> DatabaseManager:
    """Get the pumpfun database manager instance."""
    global _pumpfun_manager
    if _pumpfun_manager is None:
        _pumpfun_manager = DatabaseManager(get_pumpfun_database_url())
    return _pumpfun_manager
