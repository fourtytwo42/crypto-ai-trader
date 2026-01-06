"""Add symbol columns for multi-asset candles/features.

Revision ID: 002
Revises: 001
Create Date: 2026-01-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "candles",
        sa.Column(
            "symbol",
            sa.String(length=20),
            nullable=False,
            server_default="BTC-USDT",
        ),
    )
    op.add_column(
        "features",
        sa.Column(
            "symbol",
            sa.String(length=20),
            nullable=False,
            server_default="BTC-USDT",
        ),
    )

    op.execute("ALTER TABLE candles DROP CONSTRAINT IF EXISTS candles_timestamp_key")
    op.execute("ALTER TABLE features DROP CONSTRAINT IF EXISTS features_timestamp_key")

    op.create_unique_constraint(
        "uq_candles_symbol_timestamp",
        "candles",
        ["symbol", "timestamp"],
    )
    op.create_unique_constraint(
        "uq_features_symbol_timestamp",
        "features",
        ["symbol", "timestamp"],
    )
    op.create_index(
        "idx_candles_symbol_timestamp",
        "candles",
        ["symbol", "timestamp"],
    )
    op.create_index(
        "idx_features_symbol_timestamp",
        "features",
        ["symbol", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("idx_features_symbol_timestamp", table_name="features")
    op.drop_index("idx_candles_symbol_timestamp", table_name="candles")
    op.drop_constraint("uq_features_symbol_timestamp", "features", type_="unique")
    op.drop_constraint("uq_candles_symbol_timestamp", "candles", type_="unique")

    op.execute("ALTER TABLE candles DROP CONSTRAINT IF EXISTS candles_timestamp_key")
    op.execute("ALTER TABLE features DROP CONSTRAINT IF EXISTS features_timestamp_key")
    op.create_unique_constraint("candles_timestamp_key", "candles", ["timestamp"])
    op.create_unique_constraint("features_timestamp_key", "features", ["timestamp"])

    op.drop_column("features", "symbol")
    op.drop_column("candles", "symbol")
