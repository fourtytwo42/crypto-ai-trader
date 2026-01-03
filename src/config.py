"""Configuration management for Bitcoin Trading Model.

Loads configuration from environment variables with validation
and default values. Uses pydantic-settings for type-safe config.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql://trading_user:password@localhost:5432/bitcoin_trading",
        description="PostgreSQL connection string",
    )

    # Model Storage
    model_dir: Path = Field(default=Path("models"), description="Directory for saved models")

    # Data Storage
    data_dir: Path = Field(default=Path("data"), description="Directory for data files")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API server port")
    api_reload: bool = Field(default=True, description="Enable auto-reload for development")

    # Training Configuration
    train_device: Literal["cuda", "cpu"] = Field(default="cpu", description="Training device")
    train_batch_size: int = Field(default=32, ge=8, le=128, description="Training batch size")
    train_learning_rate: float = Field(
        default=0.0001, ge=0.00001, le=0.001, description="Learning rate"
    )
    train_epochs: int = Field(default=100, ge=1, le=500, description="Maximum training epochs")
    train_early_stopping_patience: int = Field(
        default=10, ge=1, le=50, description="Early stopping patience"
    )
    train_validation_split: float = Field(
        default=0.15, ge=0.1, le=0.3, description="Validation split ratio"
    )

    # Model Configuration
    model_type: Literal["patchtst", "nhits"] = Field(
        default="patchtst", description="Model architecture type"
    )
    model_context_length: int = Field(
        default=128, ge=32, le=512, description="Context window length in days"
    )
    model_horizon: int = Field(default=1, ge=1, le=30, description="Prediction horizon in days")
    model_hidden_size: int = Field(default=512, ge=64, le=2048, description="Hidden layer size")
    model_num_layers: int = Field(default=6, ge=1, le=24, description="Number of layers")
    model_patch_length: int = Field(
        default=16, ge=4, le=64, description="Patch length for PatchTST"
    )
    model_stride: int = Field(default=8, ge=1, le=32, description="Stride for PatchTST")

    # Signal Generation
    signal_threshold: float = Field(
        default=0.005, ge=0.001, le=0.05, description="Signal threshold (0.5% default)"
    )

    # Backtesting Configuration
    backtest_fees: float = Field(
        default=0.001, ge=0.0, le=0.01, description="Fee rate per trade (0.1% default)"
    )
    backtest_slippage: float = Field(
        default=0.0005, ge=0.0, le=0.01, description="Slippage rate (0.05% default)"
    )
    backtest_train_window: int = Field(
        default=1825, ge=365, le=3650, description="Training window in days"
    )
    backtest_test_window: int = Field(
        default=365, ge=30, le=730, description="Test window in days"
    )

    # Walk-Forward Configuration
    walk_forward_purge_days: int = Field(
        default=1, ge=0, le=30, description="Purge days between train/test"
    )
    walk_forward_embargo_days: int = Field(
        default=0, ge=0, le=30, description="Embargo days after test"
    )

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )
    log_file: Path = Field(default=Path("logs/app.log"), description="Log file path")
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")

    @field_validator("model_dir", "data_dir", "log_file", mode="before")
    @classmethod
    def validate_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    @property
    def quantiles(self) -> list[float]:
        """Default quantiles for probabilistic forecasting."""
        return [0.1, 0.5, 0.9]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
