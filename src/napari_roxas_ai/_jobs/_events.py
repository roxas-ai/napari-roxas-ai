"""
Events yielded by job operations.

Cancellation happens between two yields, so an operation should yield at
least once per file or stage.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Progress:
    """Progress of a job and an optional message."""

    current: int
    total: int
    message: str | None = None

    @property
    def fraction(self) -> float:
        """Done fraction in [0, 1] (0 when the total is unknown)."""
        if self.total <= 0:
            return 0.0
        return min(max(self.current / self.total, 0.0), 1.0)

    @property
    def percent(self) -> int:
        """Done percentage in [0, 100]."""
        return int(100 * self.fraction)


@dataclass(frozen=True)
class Message:
    """A message for the user (shown as a napari notification by default)."""

    text: str
    level: Literal["info", "warning"] = "info"


@dataclass(frozen=True)
class Partial:
    """An intermediate result."""

    payload: Any
