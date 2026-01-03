"""Model loading utilities."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelBundle:
    """Loaded model artifacts."""

    model: object
    metadata: dict[str, object]
    scaler: object | None


def load_model_artifacts(model_dir: str | Path) -> ModelBundle:
    """Load model artifacts from disk."""
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"
    scaler_path = model_dir / "scaler.pkl"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("model artifacts not found")

    model = None
    try:
        import torch

        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None

        if safe_globals is not None:
            from src.training.model_factory import SimpleQuantileModel

            with safe_globals([SimpleQuantileModel]):
                model = torch.load(model_path, weights_only=False)
        else:
            model = torch.load(model_path, weights_only=False)
    except Exception:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
    metadata = json.loads(metadata_path.read_text())

    scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)

    return ModelBundle(model=model, metadata=metadata, scaler=scaler)
