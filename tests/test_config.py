"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import Settings, get_settings


class TestSettings:
    """Test settings configuration."""

    def test_default_settings(self, monkeypatch):
        """Test default settings values."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        settings = Settings()

        # Database defaults
        assert "postgresql://" in settings.database_url

        # Model defaults
        assert settings.model_type == "patchtst"
        assert settings.model_context_length == 128
        assert settings.model_horizon == 1
        assert settings.model_hidden_size == 512
        assert settings.model_num_layers == 6

        # Training defaults
        assert settings.train_device in ["cuda", "cpu"]
        assert settings.train_batch_size == 32
        assert settings.train_learning_rate == 0.0001
        assert settings.train_epochs == 100

        # Signal defaults
        assert settings.signal_threshold == 0.005

        # Backtest defaults
        assert settings.backtest_fees == 0.001
        assert settings.backtest_slippage == 0.0005

    def test_path_conversion(self):
        """Test path fields are converted to Path objects."""
        settings = Settings(model_dir="models", data_dir="data")

        assert isinstance(settings.model_dir, Path)
        assert isinstance(settings.data_dir, Path)
        assert settings.model_dir == Path("models")
        assert settings.data_dir == Path("data")

    def test_quantiles_property(self):
        """Test quantiles property returns correct values."""
        settings = Settings()
        assert settings.quantiles == [0.1, 0.5, 0.9]

    def test_validation_batch_size(self):
        """Test batch size validation."""
        # Valid batch sizes
        settings = Settings(train_batch_size=8)
        assert settings.train_batch_size == 8

        settings = Settings(train_batch_size=128)
        assert settings.train_batch_size == 128

        # Invalid batch size should raise error
        with pytest.raises(ValueError):
            Settings(train_batch_size=1)

        with pytest.raises(ValueError):
            Settings(train_batch_size=256)

    def test_validation_learning_rate(self):
        """Test learning rate validation."""
        # Valid learning rates
        settings = Settings(train_learning_rate=0.00001)
        assert settings.train_learning_rate == 0.00001

        settings = Settings(train_learning_rate=0.001)
        assert settings.train_learning_rate == 0.001

        # Invalid learning rate should raise error
        with pytest.raises(ValueError):
            Settings(train_learning_rate=0.0)

        with pytest.raises(ValueError):
            Settings(train_learning_rate=0.01)

    def test_validation_signal_threshold(self):
        """Test signal threshold validation."""
        # Valid thresholds
        settings = Settings(signal_threshold=0.001)
        assert settings.signal_threshold == 0.001

        settings = Settings(signal_threshold=0.05)
        assert settings.signal_threshold == 0.05

        # Invalid threshold should raise error
        with pytest.raises(ValueError):
            Settings(signal_threshold=0.0001)

        with pytest.raises(ValueError):
            Settings(signal_threshold=0.1)

    def test_validation_model_type(self):
        """Test model type validation."""
        settings = Settings(model_type="patchtst")
        assert settings.model_type == "patchtst"

        settings = Settings(model_type="nhits")
        assert settings.model_type == "nhits"

        # Invalid model type should raise error
        with pytest.raises(ValueError):
            Settings(model_type="invalid")

    def test_validation_train_device(self):
        """Test train device validation."""
        settings = Settings(train_device="cuda")
        assert settings.train_device == "cuda"

        settings = Settings(train_device="cpu")
        assert settings.train_device == "cpu"

        # Invalid device should raise error
        with pytest.raises(ValueError):
            Settings(train_device="gpu")

    def test_validation_log_level(self):
        """Test log level validation."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Invalid log level should raise error
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_validation_api_port(self):
        """Test API port validation."""
        settings = Settings(api_port=8080)
        assert settings.api_port == 8080

        # Port below minimum should raise error
        with pytest.raises(ValueError):
            Settings(api_port=80)

        # Port above maximum should raise error
        with pytest.raises(ValueError):
            Settings(api_port=70000)

    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://trading_user:password@localhost:5432/bitcoin_trading",
        "MODEL_TYPE": "nhits",
        "TRAIN_DEVICE": "cpu",
    })
    def test_environment_variable_loading(self):
        """Test loading settings from environment variables."""
        settings = Settings()

        assert settings.database_url == "postgresql://trading_user:password@localhost:5432/bitcoin_trading"
        assert settings.model_type == "nhits"
        assert settings.train_device == "cpu"


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_returns_settings(self):
        """Test get_settings returns Settings instance."""
        # Clear cache first
        get_settings.cache_clear()

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Test get_settings returns cached instance."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2
