from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import get_model_dir


CLASSIFIER_FEATURE_COLS = [
    "range",
    "body",
    "dlog_volume",
    "ret_mean_5",
    "ret_std_5",
    "ret_mean_15",
    "ret_std_15",
    "ret_mean_60",
    "ret_std_60",
    "minutes_since_launch",
    "minutes_since_koth",
    "has_koth",
    "koth_reached",
    "is_completed",
    "log_trades",
    "log_close",
    "log_volume",
    "return",
]


@dataclass
class ClassifierBundle:
    model: nn.Module
    threshold: float
    horizon: int
    feature_cols: list[str]


class DirectionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_classifier(model_dir: Path) -> ClassifierBundle | None:
    meta_path = model_dir / "classifier_meta.json"
    state_path = model_dir / "classifier.pt"
    if not meta_path.exists() or not state_path.exists():
        return None

    meta = json.loads(meta_path.read_text())
    input_dim = int(meta.get("input_dim", 0))
    hidden_dim = int(meta.get("hidden_dim", 128))
    dropout = float(meta.get("dropout", 0.1))
    num_layers = int(meta.get("num_layers", 2))
    threshold = float(meta.get("threshold", 0.5))
    horizon = int(meta.get("horizon_minutes", 10))

    model = DirectionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_layers=num_layers,
    )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    feature_cols = meta.get("feature_cols") or CLASSIFIER_FEATURE_COLS
    return ClassifierBundle(model=model, threshold=threshold, horizon=horizon, feature_cols=feature_cols)


def get_classifier_bundle() -> ClassifierBundle | None:
    base_dir = get_model_dir().parent
    classifier_dir = base_dir / "classifier"
    return load_classifier(classifier_dir)


def predict_direction_confidence(features_df) -> tuple[float | None, str | None]:
    bundle = get_classifier_bundle()
    if bundle is None:
        return None, None

    feature_cols = [col for col in bundle.feature_cols if col in features_df.columns]
    if not feature_cols:
        return None, None

    last_row = features_df.tail(1)[feature_cols].to_numpy(dtype=np.float32)
    with torch.no_grad():
        logits = bundle.model(torch.from_numpy(last_row))
        prob_up = torch.sigmoid(logits).item()
    direction = "UP" if prob_up >= bundle.threshold else "DOWN"
    confidence = prob_up if direction == "UP" else 1.0 - prob_up
    return float(confidence), direction
