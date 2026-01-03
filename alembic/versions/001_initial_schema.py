"""Initial schema

Revision ID: 001
Revises:
Create Date: 2025-01-28
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create candles table
    op.create_table(
        "candles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("high", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("low", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("close", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("volume", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("timestamp"),
        sa.CheckConstraint("high >= low", name="check_high_gte_low"),
        sa.CheckConstraint("close >= low", name="check_close_gte_low"),
        sa.CheckConstraint("close <= high", name="check_close_lte_high"),
        sa.CheckConstraint("open > 0", name="check_open_positive"),
        sa.CheckConstraint("volume >= 0", name="check_volume_non_negative"),
    )
    op.create_index("idx_candles_timestamp", "candles", ["timestamp"])
    op.create_index("idx_candles_created_at", "candles", ["created_at"])

    # Create features table
    op.create_table(
        "features",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("candle_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("return", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("range", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("body", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("dlog_volume", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("ret_mean_7", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("ret_std_7", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("ret_mean_30", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("ret_std_30", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["candle_id"], ["candles.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("timestamp"),
    )
    op.create_index("idx_features_timestamp", "features", ["timestamp"])
    op.create_index("idx_features_candle_id", "features", ["candle_id"])

    # Create models table
    op.create_table(
        "models",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("version", sa.String(length=50), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("scaler_path", sa.String(length=500), nullable=True),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("train_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("train_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("val_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("val_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("test_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("idx_models_name", "models", ["name"])
    op.create_index("idx_models_type", "models", ["model_type"])
    op.create_index("idx_models_created_at", "models", ["created_at"])

    # Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("prediction_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("q10", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("q50", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("q90", sa.Numeric(precision=20, scale=8), nullable=True),
        sa.Column("signal", sa.String(length=10), nullable=True),
        sa.Column("threshold", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["model_id"], ["models.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "model_id", "prediction_timestamp", name="uq_prediction_model_timestamp"
        ),
        sa.CheckConstraint(
            "signal IN ('long', 'short', 'flat')", name="check_signal_valid"
        ),
    )
    op.create_index("idx_predictions_model_id", "predictions", ["model_id"])
    op.create_index("idx_predictions_timestamp", "predictions", ["timestamp"])
    op.create_index(
        "idx_predictions_prediction_timestamp", "predictions", ["prediction_timestamp"]
    )
    op.create_index("idx_predictions_signal", "predictions", ["signal"])

    # Create backtests table
    op.create_table(
        "backtests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("train_window", sa.Integer(), nullable=True),
        sa.Column("test_window", sa.Integer(), nullable=True),
        sa.Column("fees", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("slippage", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["model_id"], ["models.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("idx_backtests_model_id", "backtests", ["model_id"])
    op.create_index("idx_backtests_start_date", "backtests", ["start_date"])
    op.create_index("idx_backtests_end_date", "backtests", ["end_date"])

    # Create backtest_trades table
    op.create_table(
        "backtest_trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("backtest_id", sa.Integer(), nullable=False),
        sa.Column("entry_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signal", sa.String(length=10), nullable=False),
        sa.Column("entry_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("exit_price", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("fees", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("slippage", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column("net_pnl", sa.Numeric(precision=20, scale=8), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["backtest_id"], ["backtests.id"], ondelete="CASCADE"
        ),
        sa.CheckConstraint(
            "signal IN ('long', 'short')", name="check_trade_signal_valid"
        ),
    )
    op.create_index(
        "idx_backtest_trades_backtest_id", "backtest_trades", ["backtest_id"]
    )
    op.create_index(
        "idx_backtest_trades_entry_timestamp", "backtest_trades", ["entry_timestamp"]
    )
    op.create_index(
        "idx_backtest_trades_exit_timestamp", "backtest_trades", ["exit_timestamp"]
    )


def downgrade() -> None:
    op.drop_table("backtest_trades")
    op.drop_table("backtests")
    op.drop_table("predictions")
    op.drop_table("models")
    op.drop_table("features")
    op.drop_table("candles")
