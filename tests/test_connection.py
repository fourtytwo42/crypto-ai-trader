"""Tests for database connection module."""

import pytest
from sqlalchemy.engine import Engine

from src.database.connection import (
    DatabaseManager,
    create_db_engine,
    get_db_manager,
    get_session,
    get_session_factory,
)


class TestDatabaseConnection:
    """Test database connection functionality."""

    def test_create_db_engine(self, test_db_url_with_schema):
        """Test creating database engine."""
        engine = create_db_engine(test_db_url_with_schema)
        assert isinstance(engine, Engine)
        engine.dispose()

    def test_create_db_engine_from_settings(self):
        """Test creating engine using default settings."""
        # This will use the settings DATABASE_URL
        engine = create_db_engine()
        assert isinstance(engine, Engine)
        engine.dispose()

    def test_get_session_factory(self, test_db_url_with_schema):
        """Test creating session factory."""
        engine = create_db_engine(test_db_url_with_schema)
        factory = get_session_factory(engine)

        # Create a session
        session = factory()
        assert session is not None
        session.close()
        engine.dispose()


class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_init(self, test_db_url_with_schema):
        """Test DatabaseManager initialization."""
        db = DatabaseManager(test_db_url_with_schema)
        assert db._engine is None
        assert db._session_factory is None

    def test_engine_property(self, test_db_url_with_schema):
        """Test engine property creates engine on demand."""
        db = DatabaseManager(test_db_url_with_schema)
        engine = db.engine

        assert engine is not None
        assert isinstance(engine, Engine)
        db.close()

    def test_session_factory_property(self, test_db_url_with_schema):
        """Test session factory property."""
        db = DatabaseManager(test_db_url_with_schema)
        factory = db.session_factory

        assert factory is not None
        db.close()

    def test_session_context_manager(self, test_db_url_with_schema):
        """Test session context manager."""
        db = DatabaseManager(test_db_url_with_schema)

        with db.session() as session:
            assert session is not None
            # Session should be usable
            from sqlalchemy import text

            result = session.execute(text("SELECT 1"))
            assert result is not None

        db.close()

    def test_session_rollback_on_error(self, test_db_url_with_schema):
        """Test session rolls back on error."""
        db = DatabaseManager(test_db_url_with_schema)

        try:
            with db.session() as session:
                # Simulate an error
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        db.close()

    def test_test_connection_success(self, test_db_url_with_schema):
        """Test successful connection test."""
        db = DatabaseManager(test_db_url_with_schema)
        result = db.test_connection()

        assert result is True
        db.close()

    def test_test_connection_failure(self):
        """Test failed connection test."""
        db = DatabaseManager("postgresql://invalid:invalid@localhost:99999/invalid")
        result = db.test_connection()

        assert result is False

    def test_close(self, test_db_url_with_schema):
        """Test closing database connections."""
        db = DatabaseManager(test_db_url_with_schema)

        # Access engine to create it
        _ = db.engine
        assert db._engine is not None

        db.close()
        assert db._engine is None
        assert db._session_factory is None


class TestGetDbManager:
    """Test get_db_manager function."""

    def test_get_db_manager(self, test_db_url_with_schema):
        """Test getting database manager."""
        # Reset global
        import src.database.connection as conn_module
        conn_module._db_manager = None

        manager = get_db_manager(test_db_url_with_schema)
        assert manager is not None

        # Should return same instance
        manager2 = get_db_manager()
        assert manager is manager2

        manager.close()
        conn_module._db_manager = None


class TestGetSession:
    """Test get_session dependency."""

    def test_get_session(self, test_db_url_with_schema):
        """Test getting session via dependency."""
        import src.database.connection as conn_module
        conn_module._db_manager = None

        # Set up manager
        manager = get_db_manager(test_db_url_with_schema)

        # Use generator
        gen = get_session()
        session = next(gen)

        assert session is not None

        # Clean up
        try:
            next(gen)
        except StopIteration:
            pass

        manager.close()
        conn_module._db_manager = None
