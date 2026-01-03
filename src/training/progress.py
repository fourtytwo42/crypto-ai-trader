"""Training progress callbacks."""

from __future__ import annotations

import sys
from pytorch_lightning import Callback, Trainer


class EpochProgressCallback(Callback):
    """Log epoch progress with total epochs."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer: Trainer, *args: object, **kwargs: object) -> None:
        epoch = int(trainer.current_epoch) + 1
        width = 30
        filled = int(width * epoch / self.total_epochs)
        bar = "#" * filled + "-" * (width - filled)
        line = f"Training epochs [{bar}] {epoch}/{self.total_epochs}"
        end = "\n" if epoch >= self.total_epochs else "\r"
        print(line, end=end, file=sys.stdout, flush=True)
