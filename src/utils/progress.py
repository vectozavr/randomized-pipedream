from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(
        self,
        total: int,
        *,
        label: str,
        enabled: bool = True,
        width: int = 28,
        min_interval: float = 0.5,
    ) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled
        self.width = width
        self.min_interval = min_interval
        self.start = time.monotonic()
        self.last_print = 0.0
        self.current = 0

    def update(self, current: int, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        current = min(max(0, int(current)), self.total)
        self.current = current
        if not force and current < self.total and now - self.last_print < self.min_interval:
            return

        frac = current / self.total
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        if current > 0:
            eta = elapsed * (self.total - current) / current
        else:
            eta = 0.0

        sys.stderr.write(
            f"\r{self.label} [{bar}] {current}/{self.total} "
            f"({100.0 * frac:5.1f}%) elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        )
        sys.stderr.flush()
        self.last_print = now

    def close(self) -> None:
        if not self.enabled:
            return
        if self.current < self.total:
            self.update(self.total, force=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self) -> "ProgressBar":
        self.update(0, force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        elif self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"
