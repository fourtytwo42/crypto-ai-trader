"""API dependencies."""

from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from src.database.connection import get_session


def get_db() -> Generator[Session, None, None]:
    """Provide database session for FastAPI dependencies."""
    yield from get_session()
