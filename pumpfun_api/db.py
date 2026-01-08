from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import get_pumpfun_database_url

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(get_pumpfun_database_url(), pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session():
    if _SessionLocal is None:
        get_engine()
    return _SessionLocal()
