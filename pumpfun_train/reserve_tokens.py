"""Reserve token selection for evaluation.

Reserve tokens are tokens that are completely separate from training/holdout sets,
used for final evaluation to ensure models haven't overfit to holdout tokens.
"""

from __future__ import annotations

from pathlib import Path
from sqlalchemy.orm import Session

from pumpfun_train.db import get_pumpfun_db_manager
from pumpfun_train.training import select_holdout_tokens_stratified, select_tokens_by_market_cap


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
    bucketed_tokens = select_tokens_by_market_cap(session, min_rows)
    
    if exclude_training:
        # Get tokens that were used in ANY training (from any model)
        # Check holdout token files from all models to identify the training set
        training_tokens = set()
        from glob import glob
        base_dir = Path(__file__).parent
        model_patterns = [
            str(base_dir / "models" / "regression" / "h*/holdout_tokens.txt"),
            str(base_dir / "models" / "classifier" / "holdout_tokens.txt"),
            str(base_dir.parent / "experiments" / "models" / "**" / "holdout_tokens.txt"),
        ]
        
        for pattern in model_patterns:
            for holdout_path in glob(pattern):
                if Path(holdout_path).exists():
                    holdout_tokens = [
                        line.strip() for line in Path(holdout_path).read_text().splitlines() if line.strip()
                    ]
                    training_tokens.update(holdout_tokens)
        
        # Select tokens that are qualified but NOT in training/holdout sets
        filtered_bucketed = {
            bucket: [token_id for token_id in tokens if token_id not in training_tokens]
            for bucket, tokens in bucketed_tokens.items()
        }
        reserve_tokens = select_holdout_tokens_stratified(filtered_bucketed, reserve_count)
        return reserve_tokens
    else:
        # Don't exclude training tokens - just select from qualified pool
        return select_holdout_tokens_stratified(bucketed_tokens, reserve_count)


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
