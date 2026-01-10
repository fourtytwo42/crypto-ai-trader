"""Pump.fun model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sqlalchemy import func

from pumpfun_train.config_db import get_train_device
from pumpfun_train.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.models import PumpCandle1m
from pumpfun_train.config import TrainingConfig
from pumpfun_train.trainer import train_model


TOKEN_SUPPLY = 1_000_000_000
MIN_MARKET_CAP_USD = 10_000.0
CAP_BUCKETS = [
    ("10k-100k", 1e4, 1e5),
    ("100k-1m", 1e5, 1e6),
    ("1m-10m", 1e6, 1e7),
    ("10m-100m", 1e7, 1e8),
    ("100m+", 1e8, float("inf")),
]
MAX_BUCKET_RATIO = 3


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


def _cap_bucket(market_cap: float) -> str | None:
    if market_cap < MIN_MARKET_CAP_USD:
        return None
    for label, lower, upper in CAP_BUCKETS:
        if lower <= market_cap < upper:
            return label
    return CAP_BUCKETS[-1][0]


def select_tokens_by_market_cap(session, min_rows: int) -> dict[str, list[str]]:
    rows = (
        session.query(
            PumpCandle1m.token_id,
            func.count(PumpCandle1m.id).label("count"),
            func.max(PumpCandle1m.close).label("max_close"),
        )
        .group_by(PumpCandle1m.token_id)
        .having(func.count(PumpCandle1m.id) >= min_rows)
        .all()
    )
    bucketed: dict[str, list[str]] = {label: [] for label, _, _ in CAP_BUCKETS}
    for token_id, _count, max_close in rows:
        if max_close is None:
            continue
        market_cap = float(max_close) * TOKEN_SUPPLY
        bucket = _cap_bucket(market_cap)
        if bucket is None:
            continue
        bucketed[bucket].append(token_id)
    for bucket in bucketed:
        bucketed[bucket] = sorted(bucketed[bucket])
    return bucketed


def select_holdout_tokens_stratified(
    bucketed_tokens: dict[str, list[str]],
    holdout_count: int,
) -> list[str]:
    if holdout_count <= 0:
        return []
    bucket_order = [label for label, _, _ in CAP_BUCKETS]
    available = [b for b in bucket_order if bucketed_tokens.get(b)]
    if not available:
        return []

    allocations: dict[str, int] = {b: 0 for b in available}
    if holdout_count >= len(available):
        for b in available:
            allocations[b] = 1
        remaining = holdout_count - len(available)
    else:
        by_size = sorted(available, key=lambda b: len(bucketed_tokens[b]), reverse=True)
        for b in by_size[:holdout_count]:
            allocations[b] = 1
        remaining = 0

    by_size = sorted(available, key=lambda b: len(bucketed_tokens[b]), reverse=True)
    while remaining > 0:
        for b in by_size:
            if remaining <= 0:
                break
            allocations[b] += 1
            remaining -= 1

    holdout_tokens: list[str] = []
    for b in available:
        tokens = bucketed_tokens[b]
        take = min(allocations[b], len(tokens))
        if take > 0:
            holdout_tokens.extend(tokens[-take:])
    return holdout_tokens


def balance_tokens_by_bucket(bucketed_tokens: dict[str, list[str]]) -> list[str]:
    bucket_order = [label for label, _, _ in CAP_BUCKETS]
    bucketed = {b: list(bucketed_tokens.get(b, [])) for b in bucket_order}
    sizes = [len(tokens) for tokens in bucketed.values() if tokens]
    if not sizes:
        return []
    min_count = min(sizes)
    max_per_bucket = max(1, min_count * MAX_BUCKET_RATIO)
    balanced: list[str] = []
    for b in bucket_order:
        tokens = bucketed[b]
        if not tokens:
            continue
        balanced.extend(tokens[: min(len(tokens), max_per_bucket)])
    return balanced


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
    nhits_stack_types: list[str] | None = None,
    nhits_n_blocks: list[int] | None = None,
    nhits_mlp_units: list[list[int]] | None = None,
    nhits_n_pool_kernel_size: list[int] | None = None,
    nhits_n_freq_downsample: list[int] | None = None,
) -> PumpfunTrainingResult:
    if model_type == "nhits":
        # Best directional + price config from EXPERIMENTS.md (production NHITS 3/2/2).
        if nhits_stack_types is None:
            nhits_stack_types = ["identity", "identity", "identity"]
        if nhits_n_blocks is None:
            nhits_n_blocks = [3, 2, 2]
        if nhits_mlp_units is None:
            nhits_mlp_units = [[768, 768], [768, 768], [768, 768]]
        if nhits_n_pool_kernel_size is None:
            nhits_n_pool_kernel_size = [2, 2, 1]
        if nhits_n_freq_downsample is None:
            nhits_n_freq_downsample = [4, 2, 1]
    # Device config handled by get_train_device()
    db = get_pumpfun_db_manager()

    with db.session() as session:
        min_rows = context_length + horizon_minutes + 30
        bucketed_tokens = select_tokens_by_market_cap(session, min_rows)

    if not any(bucketed_tokens.values()):
        raise ValueError("No pump.fun tokens available for training")

    holdout_tokens = select_holdout_tokens_stratified(bucketed_tokens, holdout_count)
    holdout_set = set(holdout_tokens)
    train_bucketed = {
        bucket: [token for token in tokens if token not in holdout_set]
        for bucket, tokens in bucketed_tokens.items()
    }
    train_tokens = balance_tokens_by_bucket(train_bucketed)
    if not train_tokens:
        raise ValueError("No pump.fun tokens available after market cap filtering")

    from pumpfun_train.training_cache import get_cached_or_prepare

    with db.session() as session:
        training = get_cached_or_prepare(
            session,
            train_tokens,
            normalize=True,
            max_samples=None,
            batch_size=1000,
        )

    if training.df.empty:
        raise ValueError("No pump.fun training data available")

    target_col = "return"
    model_horizon = horizon_minutes
    if target_mode == "direct":
        training.df = add_return_target(training.df, horizon_minutes)
        target_col = "return_horizon"
        model_horizon = 1
    training.df = training.df.dropna().reset_index(drop=True)

    val_size = max(int(len(training.df) * 0.1), 30)  # 10% validation split
    train_split = training.df.iloc[:-val_size].reset_index(drop=True)
    val_split = training.df.iloc[-val_size:].reset_index(drop=True)

    feature_cols = [col for col in [*PUMPFUN_FEATURE_COLUMNS, "log_close", "log_volume"] if col in training.df.columns]

    config = TrainingConfig(
        model_type=model_type,
        horizon=model_horizon,
        device=get_train_device(),
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
        nhits_stack_types=nhits_stack_types,
        nhits_n_blocks=nhits_n_blocks,
        nhits_mlp_units=nhits_mlp_units,
        nhits_n_pool_kernel_size=nhits_n_pool_kernel_size,
        nhits_n_freq_downsample=nhits_n_freq_downsample,
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
