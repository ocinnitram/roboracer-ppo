"""Per-layout evaluation result and its outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

COMPLETED = "completed"
CRASHED = "crashed"
TIMEOUT = "timeout"


@dataclass
class LayoutResult:
    """What one policy run on one layout produced."""

    index: int
    n_obstacles: int
    outcome: str
    lap_times: list[float]
    steps: int
    sim_time: float
    terminal_v: float

    @property
    def success(self) -> bool:
        """True when every requested flying lap finished."""
        return self.outcome == COMPLETED

    def to_dict(self) -> dict:
        """Plain dict for the JSON summary."""
        return asdict(self)
