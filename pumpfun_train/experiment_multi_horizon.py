#!/usr/bin/env python3
"""Run multi-horizon pumpfun experiments with a single model (h=20)."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from pumpfun_train.cli.commands import (
    pumpfun_backtest_command,
    pumpfun_sync_command,
    pumpfun_train_command,
)
from pumpfun_train.data import PUMPFUN_FEATURE_COLUMNS, prepare_pumpfun_training_data
from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.forecast import forecast_next_horizon
from pumpfun_train.model_loader import load_model_artifacts


EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
MODELS_ROOT = EXPERIMENTS_DIR / "models" / "pumpfun_multi_horizon"
LEADERBOARD_PATH = EXPERIMENTS_DIR / "multi_horizon_leaderboard.json"


DEFAULT_CONFIGS = [
    {
        "name": "nhits-sum20-ctx240-lr1e4-e30-small",
        "horizon_minutes": 20,
        "context_length": 240,
        "model_type": "nhits",
        "target_mode": "sum",
        "hidden_size": 256,
        "num_layers": 2,
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "holdout_count": 12,
        "nhits_stack_types": ["identity", "identity"],
        "nhits_n_blocks": [1, 1],
        "nhits_mlp_units": [[128, 128], [128, 128]],
        "nhits_n_pool_kernel_size": [2, 1],
        "nhits_n_freq_downsample": [169, 1],
    },
    {
        "name": "nhits-sum20-ctx336-lr5e5-e50-large",
        "horizon_minutes": 20,
        "context_length": 336,
        "model_type": "nhits",
        "target_mode": "sum",
        "hidden_size": 512,
        "num_layers": 3,
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "holdout_count": 12,
        "nhits_stack_types": ["identity", "identity", "identity"],
        "nhits_n_blocks": [3, 2, 2],
        "nhits_mlp_units": [[768, 768], [768, 768], [768, 768]],
        "nhits_n_pool_kernel_size": [2, 2, 1],
        "nhits_n_freq_downsample": [4, 2, 1],
    },
]


def _next_experiment_number() -> int:
    nums: list[int] = []
    if EXPERIMENTS_DIR.exists():
        for path in EXPERIMENTS_DIR.iterdir():
            match = re.match(r"exp-(\d+)", path.name)
            if match:
                nums.append(int(match.group(1)))
    return (max(nums) if nums else 0) + 1


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _load_holdout_tokens(model_dir: Path) -> list[str]:
    path = model_dir / "holdout_tokens.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _benchmark_inference(
    model_dir: Path,
    minutes: int,
    runs: int = 3,
    warmup: int = 1,
) -> float | None:
    bundle = load_model_artifacts(model_dir)
    config_meta = bundle.metadata.get("config", {})
    horizon = int(config_meta.get("horizon", minutes))
    context_length = int(config_meta.get("context_length", 336))

    if horizon < minutes:
        return None

    holdout_tokens = _load_holdout_tokens(model_dir)
    if not holdout_tokens:
        return None

    db = get_pumpfun_db_manager()
    feature_cols: list[str] = []
    history = None

    with db.session() as session:
        for token_id in holdout_tokens:
            training = prepare_pumpfun_training_data(session, [token_id], normalize=True)
            df = training.df
            if df.empty or len(df) < context_length + minutes:
                continue
            df = df.sort_values("timestamp").reset_index(drop=True)
            history = df.iloc[-context_length:].reset_index(drop=True)
            feature_cols = [
                col
                for col in [*PUMPFUN_FEATURE_COLUMNS, "log_close", "log_volume"]
                if col in history.columns
            ]
            break

    if history is None:
        return None

    for _ in range(warmup):
        forecast_next_horizon(
            bundle.model,
            history,
            target_col="return",
            feature_cols=feature_cols,
            horizon=horizon,
        )

    start = time.perf_counter()
    for _ in range(runs):
        forecast_next_horizon(
            bundle.model,
            history,
            target_col="return",
            feature_cols=feature_cols,
            horizon=horizon,
        )
    end = time.perf_counter()
    return (end - start) / max(runs, 1)


def _load_leaderboard() -> list[dict[str, object]]:
    if not LEADERBOARD_PATH.exists():
        return []
    try:
        return json.loads(LEADERBOARD_PATH.read_text())
    except Exception:
        return []


def _save_leaderboard(entries: list[dict[str, object]]) -> None:
    LEADERBOARD_PATH.write_text(json.dumps(entries, indent=2))


def _rank_key(entry: dict[str, object]) -> tuple[float, float, float]:
    min_price = float(entry.get("min_price_accuracy", 0.0))
    avg_price = float(entry.get("avg_price_accuracy", 0.0))
    inference = float(entry.get("inference_avg_seconds", 9999.0))
    return (min_price, avg_price, -inference)


def _prune_models(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    entries_sorted = sorted(entries, key=_rank_key, reverse=True)
    kept = entries_sorted[:2]
    removed = entries_sorted[2:]

    for entry in removed:
        model_dir = Path(str(entry.get("model_dir", ""))).resolve()
        if MODELS_ROOT.resolve() in model_dir.parents and model_dir.exists():
            for path in sorted(model_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
    return kept


def _write_experiment_log(
    path: Path,
    exp_title: str,
    config: dict[str, object],
    sync_result: dict[str, int] | None,
    train_metrics: dict[str, object],
    backtests: dict[int, dict[str, object]],
    summary: dict[str, object],
) -> None:
    lines: list[str] = []
    lines.append(f"# {exp_title}")
    lines.append("")
    lines.append("## Goal")
    lines.append("Train one model with horizon=20 and evaluate price accuracy for minutes 1..20 on holdout tokens.")
    lines.append("")
    lines.append("## Sync")
    if sync_result:
        lines.append(f"- tokens={sync_result.get('tokens')} candles={sync_result.get('candles')} features={sync_result.get('features')}")
    else:
        lines.append("- skipped")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Training Metrics")
    lines.append("```json")
    lines.append(json.dumps(train_metrics, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Backtest (holdout tokens)")
    lines.append("| minutes | price_acc | dir_acc | mae | rmse | smape | samples |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for minutes in sorted(backtests):
        result = backtests[minutes]
        lines.append(
            f"| {minutes} | {result.get('price_accuracy_pct', 0):.2f} | "
            f"{result.get('direction_accuracy', 0):.2f} | "
            f"{result.get('mae', 0):.6f} | {result.get('rmse', 0):.6f} | "
            f"{result.get('smape', 0):.2f} | {result.get('samples', 0)} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("```json")
    lines.append(json.dumps(summary, indent=2))
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines))


def _run_experiment(config: dict[str, object], sync: bool, minutes: int, runs: int, warmup: int) -> dict[str, object]:
    exp_num = _next_experiment_number()
    exp_slug = _slug(config.get("name", "multi-horizon"))
    exp_id = f"exp-{exp_num:03d}"
    exp_title = f"Experiment {exp_num:03d} - pumpfun multi-horizon ({exp_slug})"

    model_dir = MODELS_ROOT / f"{exp_id}-{exp_slug}"
    model_dir.mkdir(parents=True, exist_ok=True)

    sync_result = pumpfun_sync_command() if sync else None

    train_start = datetime.now(timezone.utc)
    train_result = pumpfun_train_command(
        model_dir=str(model_dir),
        horizon_minutes=int(config["horizon_minutes"]),
        context_length=int(config["context_length"]),
        model_type=str(config["model_type"]),
        target_mode=str(config["target_mode"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        patch_length=int(config.get("patch_length", 8)),
        stride=int(config.get("stride", 4)),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        holdout_count=int(config["holdout_count"]),
        nhits_stack_types=config.get("nhits_stack_types"),
        nhits_n_blocks=config.get("nhits_n_blocks"),
        nhits_mlp_units=config.get("nhits_mlp_units"),
        nhits_n_pool_kernel_size=config.get("nhits_n_pool_kernel_size"),
        nhits_n_freq_downsample=config.get("nhits_n_freq_downsample"),
    )
    train_end = datetime.now(timezone.utc)

    backtests: dict[int, dict[str, object]] = {}
    for step in range(1, minutes + 1):
        backtests[step] = pumpfun_backtest_command(
            model_dir=str(model_dir),
            minutes=step,
            target_mode=str(config["target_mode"]),
            max_tokens=None,
            max_samples=None,
            use_reserve_tokens=False,
            show_progress=False,
            sample_stride=1,
        )

    price_accs = [float(backtests[m].get("price_accuracy_pct", 0)) for m in backtests]
    min_price = min(price_accs) if price_accs else 0.0
    avg_price = sum(price_accs) / len(price_accs) if price_accs else 0.0
    inference_avg = _benchmark_inference(model_dir, minutes=minutes, runs=runs, warmup=warmup)

    summary = {
        "exp_id": exp_id,
        "model_dir": str(model_dir),
        "holdout_tokens": len(train_result.get("holdout_tokens", [])),
        "min_price_accuracy": min_price,
        "avg_price_accuracy": avg_price,
        "inference_avg_seconds": inference_avg,
        "train_started_at": train_start.isoformat(),
        "train_finished_at": train_end.isoformat(),
    }

    exp_path = EXPERIMENTS_DIR / f"{exp_id}-pumpfun-mh-{exp_slug}.md"
    _write_experiment_log(
        exp_path,
        exp_title,
        config | {"model_dir": str(model_dir)},
        sync_result,
        train_result.get("metrics", {}),
        backtests,
        summary,
    )

    leaderboard = _load_leaderboard()
    leaderboard.append(summary)
    leaderboard = _prune_models(leaderboard)
    _save_leaderboard(leaderboard)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pumpfun multi-horizon experiments.")
    parser.add_argument("--config", type=str, help="Path to JSON list of configs.")
    parser.add_argument("--no-sync", action="store_true", help="Skip pumpfun sync.")
    parser.add_argument("--minutes", type=int, default=20, help="Max minutes to evaluate.")
    parser.add_argument("--runs", type=int, default=3, help="Inference benchmark runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Inference benchmark warmup runs.")
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS
    if args.config:
        configs = json.loads(Path(args.config).read_text())

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    for config in configs:
        _run_experiment(
            config=config,
            sync=not args.no_sync,
            minutes=args.minutes,
            runs=args.runs,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
