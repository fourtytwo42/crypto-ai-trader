"""Pump.fun direction classifier (short-horizon)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pumpfun_train.config_db import get_train_device
from sqlalchemy import select, func

from pumpfun_train.data import PUMPFUN_FEATURE_COLUMNS, add_return_target, prepare_pumpfun_training_data
from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.models import PumpCandle1m
from pumpfun_train.training import (
    balance_tokens_by_bucket,
    select_holdout_tokens_stratified,
    select_tokens_by_market_cap,
)


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
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare classifier data with memory-efficient per-token processing.
    
    Processes data in chunks and does feature engineering per-token to avoid
    loading all data into memory at once.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kwargs: x
    
    db = get_pumpfun_db_manager()
    
    # Process tokens in batches to reduce memory usage
    all_features_list: list[np.ndarray] = []
    all_labels_list: list[np.ndarray] = []
    total_samples = 0
    
    # Process tokens in chunks to avoid loading everything at once
    chunk_size = 50  # Process 50 tokens at a time (reduces memory)
    token_chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
    
    chunk_pbar = tqdm(
        token_chunks,
        desc="Processing token chunks",
        unit="chunk",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for chunk_idx, token_chunk in enumerate(chunk_pbar):
        with db.session() as session:
            # Load data for this chunk only
            training = prepare_pumpfun_training_data(
                session,
                token_chunk,
                normalize=normalize,
                max_samples=None,  # Don't limit per-chunk, limit total after
                batch_size=50,  # Smaller batches within chunk
            )
        
        if training.df.empty:
            continue
        
        # Process feature engineering per-chunk (much smaller memory footprint)
        df = add_return_target(training.df, horizon_minutes)
        df = df.sort_values(["token_id", "timestamp"]).reset_index(drop=True)
        
        # Process lags and rolling windows per token_id to reduce memory
        # This avoids creating massive intermediate DataFrames with groupby
        chunk_features_list: list[np.ndarray] = []
        chunk_labels_list: list[np.ndarray] = []
        
        for token_id in df["token_id"].unique():
            token_df = df[df["token_id"] == token_id].copy()
            
            # Add lag features
            for lag in range(1, 6):
                token_df[f"return_lag_{lag}"] = token_df["return"].shift(lag)
            
            # Add rolling features (min_periods=1 to avoid too many NaN drops)
            token_df["ret_mean_30"] = token_df["return"].rolling(window=30, min_periods=1).mean()
            token_df["ret_std_30"] = token_df["return"].rolling(window=30, min_periods=1).std()
            token_df["ret_mean_120"] = token_df["return"].rolling(window=120, min_periods=1).mean()
            token_df["ret_std_120"] = token_df["return"].rolling(window=120, min_periods=1).std()
            
            # Drop NaN rows (from lag features)
            token_df = token_df.dropna().reset_index(drop=True)
            
            if token_df.empty:
                continue
            
            if label_threshold > 0:
                token_df = token_df[token_df["return_horizon"].abs() > label_threshold].reset_index(drop=True)
            
            if token_df.empty:
                continue
            
            # Extract features and labels for this token
            token_features, _ = _build_feature_matrix(token_df, feature_cols=feature_cols)
            token_labels = (token_df["return_horizon"].to_numpy(dtype=np.float32) > 0).astype(np.float32)
            
            chunk_features_list.append(token_features)
            chunk_labels_list.append(token_labels)
        
        # Combine chunk results
        if chunk_features_list:
            chunk_features = np.vstack(chunk_features_list)
            chunk_labels = np.concatenate(chunk_labels_list)
            
            all_features_list.append(chunk_features)
            all_labels_list.append(chunk_labels)
            total_samples += len(chunk_features)
            
            chunk_pbar.set_postfix({"samples": f"{total_samples:,}"})
            
            # Check if we've reached max_samples
            if max_samples is not None and total_samples >= max_samples:
                # Trim last chunk if needed
                if total_samples > max_samples:
                    excess = total_samples - max_samples
                    chunk_features = chunk_features[:-excess]
                    chunk_labels = chunk_labels[:-excess]
                    all_features_list[-1] = chunk_features
                    all_labels_list[-1] = chunk_labels
                    total_samples = max_samples
                chunk_pbar.close()
                break
        
        # Free memory explicitly
        del training, df, chunk_features_list, chunk_labels_list
        import gc
        gc.collect()  # Force garbage collection between chunks
    
    chunk_pbar.close()
    
    if not all_features_list:
        raise ValueError("No pump.fun training data available")
    
    # Final concatenation (should be manageable now since we process in chunks)
    features = np.vstack(all_features_list)
    labels = np.concatenate(all_labels_list)
    
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
    max_samples: int | None = 10_000_000,  # Default: 10M samples max to prevent OOM
) -> PumpfunClassifierResult:
    device = get_train_device()
    db = get_pumpfun_db_manager()

    min_rows = max(min_token_samples, 0)
    with db.session() as session:
        bucketed_tokens = select_tokens_by_market_cap(session, min_rows)

    if not any(bucketed_tokens.values()):
        raise ValueError("No pump.fun tokens available for classifier training")

    holdout_tokens = select_holdout_tokens_stratified(bucketed_tokens, holdout_count)
    holdout_set = set(holdout_tokens)
    train_bucketed = {
        bucket: [token for token in tokens if token not in holdout_set]
        for bucket, tokens in bucketed_tokens.items()
    }
    train_tokens = balance_tokens_by_bucket(train_bucketed)
    if not train_tokens:
        raise ValueError("No pump.fun tokens available after market cap filtering")

    print(f"Preparing training data from {len(train_tokens)} tokens...")
    if max_samples is not None:
        print(f"Limiting to {max_samples:,} samples max to prevent memory issues")
    features, labels = _prepare_classifier_data(
        train_tokens,
        horizon_minutes,
        label_threshold=label_threshold,
        normalize=normalize_features,
        max_samples=max_samples,
    )
    if features.size == 0:
        raise ValueError("No classifier features available")
    
    print(f"Training data prepared: {len(features):,} samples, {features.shape[1]} features")

    val_size = max(int(len(features) * 0.1), 200)  # 10% validation split
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
        device
        if device != "cuda" or torch.cuda.is_available()
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

    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm not available
        def tqdm(iterable, **kwargs):
            return iterable

    model.train()
    
    # Progress bar for epochs
    epoch_pbar = tqdm(
        range(epochs),
        desc="Training epochs",
        unit="epoch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    for epoch in epoch_pbar:
        epoch_losses = []
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False,
            unit="batch",
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] loss={postfix}"
        )
        
        for batch_x, batch_y in batch_pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})
        
        batch_pbar.close()
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
    
    epoch_pbar.close()

    model.eval()
    with torch.no_grad():
        print("Validating and finding optimal threshold...")
        val_logits = model(val_x_tensor)
        val_probs = torch.sigmoid(val_logits)
        best_acc = 0.0
        best_thr = 0.5
        
        # Progress bar for threshold search
        threshold_pbar = tqdm(
            np.linspace(0.1, 0.9, 33),
            desc="Searching threshold",
            unit="test",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for thr in threshold_pbar:
            val_preds = (val_probs > thr).float()
            acc = float((val_preds == val_y_tensor).float().mean().item()) * 100.0
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)
            threshold_pbar.set_postfix({"best_acc": f"{best_acc:.2f}%", "thr": f"{best_thr:.3f}"})
        
        threshold_pbar.close()
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
