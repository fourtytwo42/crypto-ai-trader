"""Model loading utilities for pumpfun models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelBundle:
    model: object
    metadata: dict[str, object]


def _disable_model_logging(model: object) -> None:
    """Disable logging for NeuralForecast models to prevent FileExistsError during prediction."""
    try:
        from neuralforecast import NeuralForecast
        if not isinstance(model, NeuralForecast):
            return
        
        # Access the underlying models and disable their loggers
        models = getattr(model, "models", [])
        for submodel in models:
            # Set logger to False in trainer_kwargs to prevent logging during prediction
            trainer_kwargs = getattr(submodel, "trainer_kwargs", None)
            if isinstance(trainer_kwargs, dict):
                trainer_kwargs["logger"] = False
    except Exception:
        # If we can't disable logging, continue anyway
        pass


def load_model_artifacts(model_dir: str | Path) -> ModelBundle:
    """Load model artifacts from directory."""
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pt"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("model artifacts not found")

    metadata = json.loads(metadata_path.read_text())

    import torch

    map_location = None
    if not torch.cuda.is_available():
        map_location = torch.device("cpu")

    with model_path.open("rb") as handle:
        prefix = handle.read(2)
    is_zip = prefix == b"PK"

    try:
        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None

        safe_types: list[type] = []
        model_type = str(metadata.get("config", {}).get("model_type", "")).lower()
        if model_type in {"nhits", "patchtst"}:
            try:
                from neuralforecast import NeuralForecast
                from neuralforecast.models import NHITS, PatchTST

                safe_types.extend([NeuralForecast, NHITS, PatchTST])
            except Exception as nf_exc:
                if is_zip:
                    raise RuntimeError(
                        "Failed to load torch model. Install neuralforecast and torch."
                    ) from nf_exc

        if safe_globals is not None and safe_types:
            with safe_globals(safe_types):
                model = torch.load(model_path, weights_only=False, map_location=map_location)
        else:
            model = torch.load(model_path, weights_only=False, map_location=map_location)
    except Exception as exc:
        if is_zip:
            raise RuntimeError(
                "Failed to load torch model. Ensure torch and neuralforecast are installed."
            ) from exc
        raise

    # Disable logging for NeuralForecast models to prevent FileExistsError during prediction
    _disable_model_logging(model)

    return ModelBundle(model=model, metadata=metadata)

