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
from sqlalchemy import select

from src.pumpfun.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from src.pumpfun.db import get_pumpfun_db_manager
from src.pumpfun.models import PumpCandle1m
from src.pumpfun.training import select_holdout_tokens


CLASSIFIER_FEATURE_COLUMNS = [
    *PUMPFUN_FEATURE_COLUMNS,
    "log_close",
    "log_volume",
    "return",
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


def _build_feature_matrix(df) -> tuple[np.ndarray, list[str]]:
    feature_cols = [col for col in CLASSIFIER_FEATURE_COLUMNS if col in df.columns]
    if not feature_cols:
        raise ValueError("No classifier feature columns available")
    features = df[feature_cols].to_numpy(dtype=np.float32)
    return features, feature_cols


def _prepare_classifier_data(
    token_ids: list[str],
    horizon_minutes: int,
    label_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    db = get_pumpfun_db_manager()
    with db.session() as session:
        training = prepare_pumpfun_training_data(session, token_ids, normalize=True)

    if training.df.empty:
        raise ValueError("No pump.fun training data available")

    df = add_return_target(training.df, horizon_minutes)
    df = df.dropna().reset_index(drop=True)
    if label_threshold > 0:
        df = df[df["return_horizon"].abs() > label_threshold].reset_index(drop=True)
    features, _ = _build_feature_matrix(df)
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
) -> PumpfunClassifierResult:
    settings = get_settings()
    db = get_pumpfun_db_manager()

    with db.session() as session:
        tokens = session.execute(
            select(PumpCandle1m.token_id).group_by(PumpCandle1m.token_id)
        ).all()
    token_ids = [token_id for (token_id,) in tokens]

    if not token_ids:
        raise ValueError("No pump.fun tokens available for classifier training")

    holdout_tokens = select_holdout_tokens(sorted(token_ids), holdout_count)
    train_tokens = [token for token in token_ids if token not in holdout_tokens]

    features, labels = _prepare_classifier_data(train_tokens, horizon_minutes, label_threshold)
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

    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
        for thr in np.linspace(0.3, 0.7, 21):
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
    features, labels = _prepare_classifier_data(holdout_tokens, horizon_minutes, label_threshold)
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
