"""Add training jobs and forecast predictions tables.

Revision ID: 003
Revises: 002
Create Date: 2026-01-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "forecast_predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False, server_default="BTC-USDT"),
        sa.Column("horizon_hours", sa.Integer(), nullable=False, server_default="24"),
        sa.Column("data_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_close", sa.Numeric(20, 8), nullable=False),
        sa.Column("predicted_direction", sa.String(length=10), nullable=True),
        sa.Column("actual_close", sa.Numeric(20, 8), nullable=True),
        sa.Column("accuracy_pct", sa.Numeric(10, 4), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_id",
            "symbol",
            "data_timestamp",
            "horizon_hours",
            name="uq_forecast_pred_cache",
        ),
    )
    op.create_index(
        "idx_forecast_preds_model_id",
        "forecast_predictions",
        ["model_id"],
    )
    op.create_index(
        "idx_forecast_preds_symbol",
        "forecast_predictions",
        ["symbol"],
    )
    op.create_index(
        "idx_forecast_preds_target_ts",
        "forecast_predictions",
        ["target_timestamp"],
    )
    op.create_index(
        "idx_forecast_preds_predicted_at",
        "forecast_predictions",
        ["predicted_at"],
    )

    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("error", sa.String(length=500), nullable=True),
        sa.Column("model_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_training_jobs_status", "training_jobs", ["status"])
    op.create_index("idx_training_jobs_created_at", "training_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_training_jobs_created_at", table_name="training_jobs")
    op.drop_index("idx_training_jobs_status", table_name="training_jobs")
    op.drop_table("training_jobs")

    op.drop_index("idx_forecast_preds_predicted_at", table_name="forecast_predictions")
    op.drop_index("idx_forecast_preds_target_ts", table_name="forecast_predictions")
    op.drop_index("idx_forecast_preds_symbol", table_name="forecast_predictions")
    op.drop_index("idx_forecast_preds_model_id", table_name="forecast_predictions")
    op.drop_table("forecast_predictions")
