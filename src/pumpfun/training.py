"""Pump.fun model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sqlalchemy import func

from src.config import get_settings
from src.pumpfun.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from src.pumpfun.db import get_pumpfun_db_manager
from src.pumpfun.models import PumpCandle1m
from src.training.config import TrainingConfig
from src.training.trainer import train_model


@dataclass
class PumpfunTrainingResult:
    model_dir: Path
    metrics: dict[str, float]
    holdout_tokens: list[str]


def _select_tokens(session, min_rows: int) -> list[str]:
    tokens = (
        session.query(PumpCandle1m.token_id)
        .group_by(PumpCandle1m.token_id)
        .having(func.count(PumpCandle1m.id) >= min_rows)
        .all()
    )
    return [token_id for (token_id,) in tokens]


def select_holdout_tokens(token_ids: list[str], holdout_count: int) -> list[str]:
    if holdout_count <= 0 or holdout_count >= len(token_ids):
        return []
    return token_ids[-holdout_count:]


def train_pumpfun_model(
    model_dir: str | Path,
    horizon_minutes: int = 10,
    context_length: int = 336,
    model_type: str = "nhits",
    target_mode: str = "sum",
    hidden_size: int = 512,
    num_layers: int = 3,
    patch_length: int = 8,
    stride: int = 4,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    holdout_count: int = 12,
) -> PumpfunTrainingResult:
    settings = get_settings()
    db = get_pumpfun_db_manager()

    with db.session() as session:
        min_rows = context_length + horizon_minutes + 30
        token_ids = _select_tokens(session, min_rows)

    if not token_ids:
        raise ValueError("No pump.fun tokens available for training")

    holdout_tokens = select_holdout_tokens(sorted(token_ids), holdout_count)
    train_tokens = [token for token in token_ids if token not in holdout_tokens]

    with db.session() as session:
        training = prepare_pumpfun_training_data(session, train_tokens, normalize=True)

    if training.df.empty:
        raise ValueError("No pump.fun training data available")

    target_col = "return"
    model_horizon = horizon_minutes
    if target_mode == "direct":
        training.df = add_return_target(training.df, horizon_minutes)
        target_col = "return_horizon"
        model_horizon = 1
    training.df = training.df.dropna().reset_index(drop=True)

    val_size = max(int(len(training.df) * settings.train_validation_split), 30)
    train_split = training.df.iloc[:-val_size].reset_index(drop=True)
    val_split = training.df.iloc[-val_size:].reset_index(drop=True)

    feature_cols = [col for col in [*PUMPFUN_FEATURE_COLUMNS, "log_close", "log_volume"] if col in training.df.columns]

    config = TrainingConfig(
        model_type=model_type,
        horizon=model_horizon,
        device=settings.train_device,
        data_frequency="1min",
        context_length=context_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        patch_length=patch_length,
        stride=stride,
        loss_type="huber",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freq="T",
    )

    result = train_model(
        config,
        train_split,
        val_split,
        model_dir=model_dir,
        target_col=target_col,
        force_simple=False,
        feature_cols=feature_cols,
        save_artifacts=True,
        keep_best=False,
        unique_id_col="token_id",
    )

    model_dir = Path(model_dir)
    holdout_path = model_dir / "holdout_tokens.txt"
    holdout_path.write_text("\n".join(holdout_tokens))

    return PumpfunTrainingResult(
        model_dir=model_dir,
        metrics=result.metrics,
        holdout_tokens=holdout_tokens,
    )
