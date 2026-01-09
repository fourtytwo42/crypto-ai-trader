"""Reserve token selection for evaluation.

Reserve tokens are tokens that are completely separate from training/holdout sets,
used for final evaluation to ensure models haven't overfit to holdout tokens.
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.models import PumpCandle1m
from pumpfun_train.training import _select_tokens, select_holdout_tokens


RESERVE_TOKEN_FILE = Path(__file__).parent / "reserve_tokens.txt"


def select_reserve_tokens(
    session: Session,
    min_rows: int = 290,  # context_length (240) + max_horizon (20) + buffer (30)
    reserve_count: int = 50,
    exclude_training: bool = True,
) -> list[str]:
    """Select reserve tokens for evaluation.
    
    Args:
        session: Database session
        min_rows: Minimum number of rows required for a token to be eligible
        reserve_count: Number of reserve tokens to select
        exclude_training: If True, exclude tokens that were used in training
        
    Returns:
        List of token IDs selected as reserve tokens
    """
    # Get all qualified tokens
    all_qualified = _select_tokens(session, min_rows)
    
    if exclude_training:
        # Get tokens that were used in ANY training (from any model)
        # Check holdout token files from all models to identify the training set
        training_tokens = set()
        from glob import glob
        base_dir = Path(__file__).parent
        model_patterns = [
            str(base_dir / "models" / "regression" / "h*/holdout_tokens.txt"),
            str(base_dir / "models" / "classifier" / "holdout_tokens.txt"),
        ]
        
        for pattern in model_patterns:
            for holdout_path in glob(pattern):
                if Path(holdout_path).exists():
                    holdout_tokens = [
                        line.strip() for line in Path(holdout_path).read_text().splitlines() if line.strip()
                    ]
                    training_tokens.update(holdout_tokens)
        
        # Get ALL tokens that meet min_rows requirement
        all_tokens_stmt = (
            session.query(PumpCandle1m.token_id, func.count(PumpCandle1m.id).label('count'))
            .group_by(PumpCandle1m.token_id)
            .having(func.count(PumpCandle1m.id) >= min_rows)
        ).all()
        
        all_token_ids = [t[0] for t in all_tokens_stmt]
        all_token_counts = {t[0]: t[1] for t in all_tokens_stmt}
        
        # Select tokens that are qualified but NOT in training/holdout sets
        reserve_candidates = [
            (token_id, all_token_counts[token_id])
            for token_id in all_token_ids
            if token_id not in training_tokens
        ]
        
        # Sort by count descending (prefer tokens with more data)
        reserve_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top reserve_count tokens
        if len(reserve_candidates) >= reserve_count:
            return [token_id for token_id, _ in reserve_candidates[:reserve_count]]
        else:
            # Not enough reserve candidates - use what we have
            return [token_id for token_id, _ in reserve_candidates]
    else:
        # Don't exclude training tokens - just select from qualified pool
        # This would select tokens that could have been in training but weren't
        sorted_tokens = sorted(all_qualified)
        # Take tokens from a different part of the sorted list (e.g., middle section)
        # to ensure they're different from holdout tokens (which are last N)
        if len(sorted_tokens) > reserve_count:
            # Take from the middle section (between training and holdout)
            start_idx = len(sorted_tokens) // 3
            end_idx = start_idx + reserve_count
            if end_idx > len(sorted_tokens):
                end_idx = len(sorted_tokens)
            return sorted_tokens[start_idx:end_idx]
        return sorted_tokens[:reserve_count]


def get_reserve_tokens() -> list[str]:
    """Load reserve tokens from file, or select them if file doesn't exist."""
    if RESERVE_TOKEN_FILE.exists():
        return [
            line.strip()
            for line in RESERVE_TOKEN_FILE.read_text().splitlines()
            if line.strip()
        ]
    
    # File doesn't exist - select reserve tokens
    db = get_pumpfun_db_manager()
    with db.session() as session:
        reserve_tokens = select_reserve_tokens(session, reserve_count=50, exclude_training=True)
    
    # Save to file for consistency
    RESERVE_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESERVE_TOKEN_FILE.write_text("\n".join(reserve_tokens))
    
    return reserve_tokens


def save_reserve_tokens(token_ids: list[str]) -> None:
    """Save reserve tokens to file."""
    RESERVE_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESERVE_TOKEN_FILE.write_text("\n".join(token_ids))


def refresh_reserve_tokens(reserve_count: int = 50) -> list[str]:
    """Refresh/reselect reserve tokens from database."""
    from pumpfun_train.reserve_cache import invalidate_reserve_cache
    
    db = get_pumpfun_db_manager()
    with db.session() as session:
        reserve_tokens = select_reserve_tokens(session, reserve_count=reserve_count, exclude_training=True)
    
    save_reserve_tokens(reserve_tokens)
    
    # Invalidate reserve cache since tokens changed
    invalidate_reserve_cache()
    
    return reserve_tokens

