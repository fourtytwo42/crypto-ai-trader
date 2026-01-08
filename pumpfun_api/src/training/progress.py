"""Shim for EpochProgressCallback to satisfy model unpickling."""

from __future__ import annotations


class EpochProgressCallback:
    state_key = "EpochProgressCallback"

    def __init__(self, *_args, **_kwargs) -> None:
        self.state_key = "EpochProgressCallback"

    def setup(self, *_args, **_kwargs) -> None:
        pass

    def on_exception(self, *_args, **_kwargs) -> None:
        pass

    def on_predict_start(self, *_args, **_kwargs) -> None:
        pass

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None

        return _noop
