from src.pumpfun.training import select_holdout_tokens


def test_select_holdout_tokens():
    tokens = [f"token_{i}" for i in range(20)]
    holdout = select_holdout_tokens(tokens, 5)
    assert holdout == tokens[-5:]
