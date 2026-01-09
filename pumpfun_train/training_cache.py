"""Training data cache system.

Caches prepared training data to disk to avoid reloading from database
for each model training run. Cache is invalidated when database is updated.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from pumpfun_train.data import PumpfunTrainingData
    from pumpfun_train.models import PumpCandle1m, PumpFeature1m


CACHE_DIR = Path(__file__).parent / ".training_cache"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_DATA_FILE = CACHE_DIR / "training_data.pkl"
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"


@dataclass
class CacheMetadata:
    """Metadata about cached training data."""

    cached_at: str  # ISO timestamp
    max_feature_timestamp: str | None  # ISO timestamp of latest feature
    total_rows: int
    token_count: int
    cache_version: str = "1.0"  # For future cache format changes


def _get_database_metadata(session: Session) -> dict:
    """Get current database metadata to compare with cache."""
    from pumpfun_train.models import PumpCandle1m, PumpFeature1m

    # Get max timestamp from features table (most up-to-date indicator)
    max_feature_stmt = select(func.max(PumpFeature1m.timestamp))
    max_feature_ts = session.execute(max_feature_stmt).scalar_one_or_none()

    # Get total row count from features table
    count_stmt = select(func.count(PumpFeature1m.id))
    total_rows = session.execute(count_stmt).scalar_one_or_none() or 0

    # Get unique token count
    token_count_stmt = select(func.count(func.distinct(PumpFeature1m.token_id)))
    token_count = session.execute(token_count_stmt).scalar_one_or_none() or 0

    return {
        "max_feature_timestamp": (
            max_feature_ts.isoformat() if max_feature_ts is not None else None
        ),
        "total_rows": total_rows,
        "token_count": token_count,
    }


def load_cache_metadata() -> CacheMetadata | None:
    """Load cache metadata from disk."""
    if not CACHE_METADATA_FILE.exists():
        return None

    try:
        with open(CACHE_METADATA_FILE, "r") as f:
            data = json.load(f)
        return CacheMetadata(**data)
    except Exception:
        return None


def save_cache_metadata(metadata: CacheMetadata) -> None:
    """Save cache metadata to disk."""
    with open(CACHE_METADATA_FILE, "w") as f:
        json.dump(
            {
                "cached_at": metadata.cached_at,
                "max_feature_timestamp": metadata.max_feature_timestamp,
                "total_rows": metadata.total_rows,
                "token_count": metadata.token_count,
                "cache_version": metadata.cache_version,
            },
            f,
            indent=2,
        )


def is_cache_valid(session: Session) -> bool:
    """Check if cached training data is still valid."""
    cached_meta = load_cache_metadata()
    if cached_meta is None:
        return False

    if not CACHE_DATA_FILE.exists():
        return False

    # Get current database metadata
    db_meta = _get_database_metadata(session)

    # Cache is invalid if:
    # 1. Database has more rows (new data added)
    # 2. Database has newer timestamps
    # 3. Token count changed
    if db_meta["total_rows"] != cached_meta.total_rows:
        return False

    if db_meta["token_count"] != cached_meta.token_count:
        return False

    # Compare timestamps
    if cached_meta.max_feature_timestamp is None:
        # Cache has no timestamp, but DB might - invalidate
        if db_meta["max_feature_timestamp"] is not None:
            return False
    else:
        if db_meta["max_feature_timestamp"] is None:
            # DB has no timestamp but cache does - this shouldn't happen, invalidate
            return False
        # Compare timestamps - if DB is newer, cache is invalid
        cached_ts = datetime.fromisoformat(cached_meta.max_feature_timestamp)
        db_ts = datetime.fromisoformat(db_meta["max_feature_timestamp"])
        if db_ts > cached_ts:
            return False

    return True


def load_cached_training_data() -> PumpfunTrainingData | None:
    """Load cached training data from disk."""
    if not CACHE_DATA_FILE.exists():
        return None

    try:
        with open(CACHE_DATA_FILE, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception:
        return None


def save_training_data_cache(training_data: PumpfunTrainingData, session: Session) -> None:
    """Save training data to cache with metadata."""
    # Save the training data
    with open(CACHE_DATA_FILE, "wb") as f:
        pickle.dump(training_data, f)

    # Save metadata
    db_meta = _get_database_metadata(session)
    metadata = CacheMetadata(
        cached_at=datetime.now(timezone.utc).isoformat(),
        max_feature_timestamp=db_meta["max_feature_timestamp"],
        total_rows=db_meta["total_rows"],
        token_count=db_meta["token_count"],
    )
    save_cache_metadata(metadata)


def invalidate_cache() -> None:
    """Invalidate the cache by deleting cache files."""
    if CACHE_DATA_FILE.exists():
        CACHE_DATA_FILE.unlink()
    if CACHE_METADATA_FILE.exists():
        CACHE_METADATA_FILE.unlink()


def get_cached_or_prepare(
    session: Session,
    token_ids: list[str],
    normalize: bool = True,
    max_samples: int | None = None,
    batch_size: int = 1000,
) -> PumpfunTrainingData:
    """Get cached training data if valid, otherwise prepare and cache it.
    
    Note: Cache is based on database state (timestamp, row count), not token_ids.
    If database hasn't changed, cached data is reused regardless of which tokens
    are requested. This assumes training uses all available tokens.
    
    Args:
        session: Database session
        token_ids: List of token IDs to include in training data
        normalize: Whether to normalize features
        max_samples: Maximum number of samples to load
        batch_size: Batch size for processing tokens
        
    Returns:
        PumpfunTrainingData object
    """
    from pumpfun_train.data import prepare_pumpfun_training_data

    # Try to load from cache (cache validity is based on DB state, not token_ids)
    if is_cache_valid(session):
        cached_data = load_cached_training_data()
        if cached_data is not None:
            meta = load_cache_metadata()
            if meta:
                print(f"✓ Using cached training data (saved at {meta.cached_at}, {meta.total_rows:,} rows, {meta.token_count} tokens)")
                return cached_data

    # Cache invalid or doesn't exist - prepare from database
    print("Loading training data from database (this may take a few minutes)...")
    training_data = prepare_pumpfun_training_data(
        session,
        token_ids,
        normalize=normalize,
        max_samples=max_samples,
        batch_size=batch_size,
    )

    # Save to cache for next time
    if not training_data.df.empty:
        save_training_data_cache(training_data, session)
        meta = load_cache_metadata()
        if meta:
            print(f"✓ Training data cached ({meta.total_rows:,} rows, {meta.token_count} tokens)")

    return training_data

