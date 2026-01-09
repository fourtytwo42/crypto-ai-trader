"""Training progress callbacks."""

from __future__ import annotations

import sys
import time
import structlog
from pytorch_lightning import Callback, Trainer

logger = structlog.get_logger(__name__)

_ACTIVE_DISPLAY: "ProgressDisplay | None" = None


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "ETA --"
    minutes, sec = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"ETA {hours}h{minutes:02d}m"
    if minutes:
        return f"ETA {minutes}m{sec:02d}s"
    return f"ETA {sec}s"


def _render_progress_line(
    label: str,
    current: int,
    total: int,
    *,
    width: int = 24,
    suffix: str = "",
) -> str:
    total = max(total, 1)
    current = min(max(current, 0), total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    if suffix:
        return f"{label} [{bar}] {current}/{total} | {suffix}"
    return f"{label} [{bar}] {current}/{total}"


class ProgressDisplay:
    """Persistent two-line progress display with ETA."""

    def __init__(
        self,
        total_windows: int,
        total_epochs: int,
        *,
        window_label: str = "Forecast windows",
        epoch_label: str = "Training epochs",
    ) -> None:
        self.total_windows = max(total_windows, 1)
        self.total_epochs = max(total_epochs, 1)
        self.window_label = window_label
        self.epoch_label = epoch_label
        self.current_window = 0
        self.current_epoch = 0
        self._printed = False
        self._window_start: float | None = None
        self._epoch_last: float | None = None
        self._avg_epoch: float | None = None
        self._avg_window: float | None = None

    def set_window(self, current: int, total: int | None = None) -> None:
        if total is not None:
            self.total_windows = max(total, 1)
        self.current_window = max(current, 0)
        self._window_start = time.monotonic()
        self._epoch_last = None
        self._avg_epoch = None
        self._render()

    def update_epoch(self, current: int, total: int | None = None) -> None:
        if total is not None:
            self.total_epochs = max(total, 1)
        now = time.monotonic()
        if self._epoch_last is not None:
            elapsed = now - self._epoch_last
            if elapsed > 0:
                if self._avg_epoch is None:
                    self._avg_epoch = elapsed
                else:
                    self._avg_epoch = 0.8 * self._avg_epoch + 0.2 * elapsed
        self._epoch_last = now
        self.current_epoch = max(current, 0)
        self._render()

    def finish_window(self) -> None:
        if self._window_start is None:
            return
        elapsed = time.monotonic() - self._window_start
        if elapsed > 0:
            if self._avg_window is None:
                self._avg_window = elapsed
            else:
                self._avg_window = 0.8 * self._avg_window + 0.2 * elapsed
        self._render()

    def _render(self) -> None:
        remaining_epochs = max(self.total_epochs - self.current_epoch, 0)
        epoch_eta = self._avg_epoch * remaining_epochs if self._avg_epoch else None
        epoch_suffix_parts = [_format_eta(epoch_eta)]
        if self._avg_epoch:
            epoch_suffix_parts.append(f"{self._avg_epoch:.1f}s/ep")
        epoch_suffix = " | ".join(epoch_suffix_parts)
        epoch_line = _render_progress_line(
            self.epoch_label,
            self.current_epoch,
            self.total_epochs,
            suffix=epoch_suffix,
        )

        remaining_windows = max(self.total_windows - self.current_window, 0)
        window_eta = self._avg_window * remaining_windows if self._avg_window else None
        window_line = _render_progress_line(
            self.window_label,
            self.current_window,
            self.total_windows,
            suffix=_format_eta(window_eta),
        )

        try:
            if self._printed:
                sys.stdout.write("\x1b[2A")
                sys.stdout.write("\x1b[2K\r" + window_line + "\n")
                sys.stdout.write("\x1b[2K\r" + epoch_line + "\n")
            else:
                sys.stdout.write(window_line + "\n" + epoch_line + "\n")
                self._printed = True
            sys.stdout.flush()
        except OSError:
            if not self._printed:
                logger.info(
                    "Progress",
                    window=f"{self.current_window}/{self.total_windows}",
                    epoch=f"{self.current_epoch}/{self.total_epochs}",
                )
                self._printed = True


def set_active_display(display: ProgressDisplay | None) -> None:
    global _ACTIVE_DISPLAY
    _ACTIVE_DISPLAY = display


def get_active_display() -> ProgressDisplay | None:
    return _ACTIVE_DISPLAY


class EpochProgressCallback(Callback):
    """Log epoch progress with total epochs."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs

    def on_train_epoch_end(self, trainer: Trainer, *args: object, **kwargs: object) -> None:
        epoch = int(trainer.current_epoch) + 1
        display = get_active_display()
        if display:
            display.update_epoch(epoch, total=self.total_epochs)
            return
        line = _render_progress_line(
            "Training epochs",
            epoch,
            self.total_epochs,
        )
        end = "\n" if epoch >= self.total_epochs else "\r"
        try:
            print(line, end=end, file=sys.stdout, flush=True)
        except OSError:
            if end == "\n":
                logger.info("Training epochs", epoch=epoch, total_epochs=self.total_epochs)
