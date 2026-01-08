"""Pump.fun direction classifier (short-horizon)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_settings
from sqlalchemy import select, func

from src.pumpfun.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from src.pumpfun.db import get_pumpfun_db_manager
from src.pumpfun.models import PumpCandle1m
from src.pumpfun.training import select_holdout_tokens


CLASSIFIER_FEATURE_COLUMNS = [
    *PUMPFUN_FEATURE_COLUMNS,
    "log_close",
    "log_volume",
    "return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_4",
    "return_lag_5",
    "ret_mean_30",
    "ret_std_30",
    "ret_mean_120",
    "ret_std_120",
]


@dataclass
class PumpfunClassifierResult:
    model_dir: Path
    metrics: dict[str, float]
    holdout_tokens: list[str]


class PumpfunDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _build_feature_matrix(df, feature_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    base_cols = feature_cols or CLASSIFIER_FEATURE_COLUMNS
    feature_cols = [col for col in base_cols if col in df.columns]
    if not feature_cols:
        raise ValueError("No classifier feature columns available")
    features = df[feature_cols].to_numpy(dtype=np.float32)
    return features, feature_cols


def _prepare_classifier_data(
    token_ids: list[str],
    horizon_minutes: int,
    label_threshold: float = 0.0,
    normalize: bool = True,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    db = get_pumpfun_db_manager()
    with db.session() as session:
        training = prepare_pumpfun_training_data(session, token_ids, normalize=normalize)

    if training.df.empty:
        raise ValueError("No pump.fun training data available")

    df = add_return_target(training.df, horizon_minutes)
    df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
    for lag in range(1, 6):
        df[f"return_lag_{lag}"] = df.groupby("token_id")["return"].shift(lag)
    df["ret_mean_30"] = df.groupby("token_id")["return"].rolling(window=30).mean().reset_index(level=0, drop=True)
    df["ret_std_30"] = df.groupby("token_id")["return"].rolling(window=30).std().reset_index(level=0, drop=True)
    df["ret_mean_120"] = df.groupby("token_id")["return"].rolling(window=120).mean().reset_index(level=0, drop=True)
    df["ret_std_120"] = df.groupby("token_id")["return"].rolling(window=120).std().reset_index(level=0, drop=True)
    df = df.dropna().reset_index(drop=True)
    if label_threshold > 0:
        df = df[df["return_horizon"].abs() > label_threshold].reset_index(drop=True)
    features, _ = _build_feature_matrix(df, feature_cols=feature_cols)
    labels = (df["return_horizon"].to_numpy(dtype=np.float32) > 0).astype(np.float32)
    return features, labels


def train_pumpfun_direction_classifier(
    model_dir: str | Path,
    horizon_minutes: int = 10,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    label_threshold: float = 0.0,
    holdout_count: int = 12,
    min_token_samples: int = 0,
    use_pos_weight: bool = True,
    normalize_features: bool = True,
) -> PumpfunClassifierResult:
    settings = get_settings()
    db = get_pumpfun_db_manager()

    with db.session() as session:
        tokens = session.execute(
            select(PumpCandle1m.token_id, func.count(PumpCandle1m.token_id))
            .group_by(PumpCandle1m.token_id)
        ).all()
    token_ids = [
        token_id for (token_id, count) in tokens if count >= max(min_token_samples, 0)
    ]

    if not token_ids:
        raise ValueError("No pump.fun tokens available for classifier training")

    holdout_tokens = select_holdout_tokens(sorted(token_ids), holdout_count)
    train_tokens = [token for token in token_ids if token not in holdout_tokens]

    features, labels = _prepare_classifier_data(
        train_tokens,
        horizon_minutes,
        label_threshold=label_threshold,
        normalize=normalize_features,
    )
    if features.size == 0:
        raise ValueError("No classifier features available")

    val_size = max(int(len(features) * settings.train_validation_split), 200)
    train_x = features[:-val_size]
    train_y = labels[:-val_size]
    val_x = features[-val_size:]
    val_y = labels[-val_size:]

    input_dim = train_x.shape[1]
    model = PumpfunDirectionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
    )
    device = torch.device(
        settings.train_device
        if settings.train_device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)

    if use_pos_weight:
        pos = float(train_y.sum())
        neg = float(len(train_y) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x_tensor = torch.from_numpy(val_x).to(device)
    val_y_tensor = torch.from_numpy(val_y).to(device)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(val_x_tensor)
        val_probs = torch.sigmoid(val_logits)
        best_acc = 0.0
        best_thr = 0.5
        for thr in np.linspace(0.1, 0.9, 33):
            val_preds = (val_probs > thr).float()
            acc = float((val_preds == val_y_tensor).float().mean().item()) * 100.0
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)
        val_acc = best_acc

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "classifier.pt")
    meta = {
        "horizon_minutes": horizon_minutes,
        "feature_cols": CLASSIFIER_FEATURE_COLUMNS,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "num_layers": num_layers,
        "label_threshold": label_threshold,
        "threshold": best_thr,
        "min_token_samples": min_token_samples,
        "use_pos_weight": use_pos_weight,
        "normalize_features": normalize_features,
    }
    (model_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=2))
    (model_dir / "holdout_tokens.txt").write_text("\n".join(holdout_tokens))

    return PumpfunClassifierResult(
        model_dir=model_dir,
        metrics={"directional_accuracy": val_acc},
        holdout_tokens=holdout_tokens,
    )


def backtest_pumpfun_direction_classifier(
    model_dir: str | Path,
    max_tokens: int | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "classifier_meta.json").read_text())
    horizon_minutes = int(meta["horizon_minutes"])
    threshold = float(meta.get("threshold", 0.5))
    label_threshold = float(meta.get("label_threshold", 0.0))

    holdout_tokens = [
        line.strip() for line in (model_dir / "holdout_tokens.txt").read_text().splitlines() if line.strip()
    ]
    if max_tokens is not None:
        holdout_tokens = holdout_tokens[:max_tokens]
    normalize_features = bool(meta.get("normalize_features", True))
    feature_cols = meta.get("feature_cols")
    features, labels = _prepare_classifier_data(
        holdout_tokens,
        horizon_minutes,
        label_threshold=label_threshold,
        normalize=normalize_features,
        feature_cols=feature_cols,
    )
    if features.size == 0:
        raise ValueError("No holdout features available")
    if max_samples is not None:
        features = features[:max_samples]
        labels = labels[:max_samples]

    input_dim = int(meta["input_dim"])
    model = PumpfunDirectionClassifier(
        input_dim=input_dim,
        hidden_dim=int(meta["hidden_dim"]),
        dropout=float(meta["dropout"]),
        num_layers=int(meta.get("num_layers", 2)),
    )
    state = torch.load(model_dir / "classifier.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float().numpy()
    accuracy = float((preds == labels).mean() * 100.0)

    return {"directional_accuracy": accuracy, "samples": float(len(labels))}
