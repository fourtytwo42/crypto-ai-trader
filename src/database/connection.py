"""Database connection management.

Provides database connection pool and session management
using SQLAlchemy 2.0 async patterns.
"""

from contextlib import contextmanager
from typing import Generator

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings

logger = structlog.get_logger(__name__)


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create SQLAlchemy database engine.

    Args:
        database_url: Optional database URL. If not provided, uses settings.

    Returns:
        SQLAlchemy Engine instance.
    """
    url = database_url or get_settings().database_url
    logger.info("Creating database engine", url=url.split("@")[-1] if "@" in url else url)

    engine_kwargs = {
        "pool_pre_ping": True,
        "echo": False,
        "pool_size": 5,
        "max_overflow": 10,
    }

    engine = create_engine(url, **engine_kwargs)
    return engine


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create session factory for database operations.

    Args:
        engine: SQLAlchemy engine.

    Returns:
        Session factory.
    """
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str | None = None) -> None:
        """Initialize database manager.

        Args:
            database_url: Optional database URL override.
        """
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._database_url = database_url

    @property
    def engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            self._engine = create_db_engine(self._database_url)
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Get or create session factory."""
        if self._session_factory is None:
            self._session_factory = get_session_factory(self.engine)
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions.

        Yields:
            Database session.

        Example:
            with db.session() as session:
                session.query(Model).all()
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        """Test database connection.

        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
            return True
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            return False

    def close(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager(database_url: str | None = None) -> DatabaseManager:
    """Get or create global database manager.

    Args:
        database_url: Optional database URL override.

    Returns:
        DatabaseManager instance.
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(database_url)
    return _db_manager


def get_session() -> Generator[Session, None, None]:
    """Dependency for getting database sessions.

    Yields:
        Database session.
    """
    db = get_db_manager()
    with db.session() as session:
        yield session
