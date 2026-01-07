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

    metadata = json.loads(metadata_path.read_text())
    model = None
    with model_path.open("rb") as handle:
        prefix = handle.read(2)
    is_zip = prefix == b"PK"

    try:
        import torch

        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None

        safe_types: list[type] = []
        try:
            from src.training.model_factory import SimpleQuantileModel

            safe_types.append(SimpleQuantileModel)
        except Exception:
            pass

        model_type = str(metadata.get("config", {}).get("model_type", "")).lower()
        if model_type in {"nhits", "patchtst"}:
            try:
                from neuralforecast import NeuralForecast
                from neuralforecast.models import NHITS, PatchTST

                safe_types.extend([NeuralForecast, NHITS, PatchTST])
            except Exception as nf_exc:
                if is_zip:
                    raise RuntimeError(
                        "Failed to load torch model. Install neuralforecast and torch, "
                        "or run with the project venv."
                    ) from nf_exc

        if safe_globals is not None and safe_types:
            with safe_globals(safe_types):
                model = torch.load(model_path, weights_only=False)
        else:
            model = torch.load(model_path, weights_only=False)
    except Exception as exc:
        if is_zip:
            raise RuntimeError(
                "Failed to load torch model. Ensure torch and neuralforecast dependencies "
                "are installed (use the project venv)."
            ) from exc
        with model_path.open("rb") as handle:
            model = pickle.load(handle)

    scaler = None
    if scaler_path.exists():
        with scaler_path.open("rb") as handle:
            scaler = pickle.load(handle)

    return ModelBundle(model=model, metadata=metadata, scaler=scaler)
