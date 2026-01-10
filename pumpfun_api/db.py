from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import get_pumpfun_database_url

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        # Limit connection pool to prevent memory issues
        # pool_size: number of connections to maintain
        # max_overflow: additional connections beyond pool_size
        _engine = create_engine(
            get_pumpfun_database_url(),
            pool_pre_ping=True,
            pool_size=5,  # Limit concurrent connections
            max_overflow=10,  # Allow some overflow but not unlimited
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session():
    if _SessionLocal is None:
        get_engine()
    return _SessionLocal()
