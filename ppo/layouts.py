"""Obstacle layouts for training and evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ppo.config import ObstaclesConfig
from sim.track import TRACKS

LAYOUTS = Path(__file__).resolve().parents[1] / "evaluation" / "layouts"


@dataclass(frozen=True)
class Obstacle:
    """One axis-aligned track-frame rectangle."""

    s: float
    n: float
    half_length: float
    half_width: float


Layout = tuple[Obstacle, ...]


@dataclass(frozen=True)
class Zones:
    """Station intervals where obstacles may be placed."""

    intervals: tuple[tuple[float, float], ...]

    @classmethod
    def load(cls, map_name: str) -> Zones:
        """Read `valid_zones_s` from `tracks/<map>/zones.json`."""
        path = TRACKS / map_name / "zones.json"
        with path.open() as handle:
            raw = json.load(handle)["valid_zones_s"]
        if not raw:
            raise ValueError(f"{path} has no placement zones")
        return cls(tuple((float(lo), float(hi)) for lo, hi in raw))


@dataclass(frozen=True)
class Track:
    """Racing-line samples used to place and plot obstacles."""

    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    w_l: np.ndarray
    w_r: np.ndarray
    n_x: np.ndarray
    n_y: np.ndarray
    kappa: np.ndarray
    length: float

    @classmethod
    def load(cls, map_name: str) -> Track:
        """Load `tracks/<map>/raceline.csv`."""
        data = np.loadtxt(TRACKS / map_name / "raceline.csv", delimiter=",", skiprows=1)
        s = data[:, 2]
        return cls(
            s=s,
            x=data[:, 0],
            y=data[:, 1],
            yaw=data[:, 4],
            w_l=data[:, 6],
            w_r=data[:, 7],
            n_x=data[:, 8],
            n_y=data[:, 9],
            kappa=data[:, 5],
            length=float(s[-1]),
        )

    def nearest_index(self, s: float) -> int:
        """Return the sample index closest to station `s`."""
        return int(np.argmin(np.abs(self.s - s)))

    def pose(self, s: float, n: float) -> tuple[float, float, float]:
        """Return world `(x, y, yaw)` of a track-frame point."""
        ndx = self.nearest_index(s)
        x = float(self.x[ndx] + n * self.n_x[ndx])
        y = float(self.y[ndx] + n * self.n_y[ndx])
        return x, y, float(self.yaw[ndx])


def _sample_n(
    track: Track,
    s: float,
    half_width: float,
    centre_deadzone: float,
    rng: np.random.Generator,
) -> float | None:
    """Draw a lateral offset inside the local width, skipping the centre band."""
    ndx = track.nearest_index(s)
    n_lo = float(-track.w_r[ndx] + half_width)
    n_hi = float(track.w_l[ndx] - half_width)
    if n_lo > n_hi:
        return None
    dead = float(centre_deadzone)
    if dead <= 0.0:
        return float(rng.uniform(n_lo, n_hi))
    intervals: list[tuple[float, float]] = []
    left_hi = min(n_hi, -dead)
    right_lo = max(n_lo, dead)
    if n_lo < left_hi:
        intervals.append((n_lo, left_hi))
    if right_lo < n_hi:
        intervals.append((right_lo, n_hi))
    if not intervals:
        return None
    lengths = np.array([hi - lo for lo, hi in intervals], dtype=float)
    pick = int(rng.choice(len(intervals), p=lengths / lengths.sum()))
    lo, hi = intervals[pick]
    return float(rng.uniform(lo, hi))


def _sample_s_from_zones(
    intervals: Sequence[tuple[float, float]],
    num_obstacles: int,
    min_s_distance: float,
    rng: np.random.Generator,
) -> list[float]:
    """Place one station per zone first, then fill remaining slots by zone length."""
    zone_lengths = np.array([s_max - s_min for s_min, s_max in intervals], dtype=float)
    if zone_lengths.size == 0 or float(zone_lengths.sum()) <= 0.0:
        return []
    zone_weights = zone_lengths / zone_lengths.sum()
    n_zones = len(intervals)
    placed: list[float] = []
    max_attempts = 100
    zone_order = [int(i) for i in rng.permutation(n_zones)]
    for zone_idx in zone_order[: min(num_obstacles, n_zones)]:
        s_min, s_max = intervals[zone_idx]
        for _ in range(max_attempts):
            s_candidate = float(rng.uniform(s_min, s_max))
            if all(abs(s_candidate - s_prev) >= min_s_distance for s_prev in placed):
                placed.append(s_candidate)
                break
    for _ in range(num_obstacles - len(placed)):
        for _ in range(max_attempts):
            zone_idx = int(rng.choice(n_zones, p=zone_weights))
            s_min, s_max = intervals[zone_idx]
            s_candidate = float(rng.uniform(s_min, s_max))
            if all(abs(s_candidate - s_prev) >= min_s_distance for s_prev in placed):
                placed.append(s_candidate)
                break
    placed.sort()
    return placed


def sample_layout(
    track: Track,
    obstacles: ObstaclesConfig,
    zones: Zones,
    rng: np.random.Generator,
) -> Layout:
    """Draw a random layout on `track` from the obstacle numbers and zones."""
    half_length = float(obstacles.obstacle_length) / 2.0
    half_width = float(obstacles.obstacle_width) / 2.0
    n_want = int(obstacles.n_obstacles)
    if n_want <= 0:
        return ()
    clipped: list[tuple[float, float]] = []
    for s_min, s_max in zones.intervals:
        lo = max(float(s_min), float(obstacles.start_exclusion))
        hi = min(float(s_max), float(track.length))
        if hi >= lo:
            clipped.append((lo, hi))
    s_values = _sample_s_from_zones(clipped, n_want, float(obstacles.min_s_distance), rng)
    items: list[Obstacle] = []
    for s in s_values:
        n = _sample_n(track, s, half_width, float(obstacles.centre_deadzone), rng)
        if n is None:
            continue
        items.append(Obstacle(s=float(s), n=float(n), half_length=half_length, half_width=half_width))
    items.sort(key=lambda item: item.s)
    return tuple(items)


def to_sim(layout: Layout) -> list[dict[str, float]]:
    """Pack a layout for `VecEnv.set_obstacles`."""
    return [
        {
            "s": item.s,
            "n": item.n,
            "half_length": item.half_length,
            "half_width": item.half_width,
        }
        for item in layout
    ]


def _index(path: Path) -> int:
    """Parse the integer index from a `k<idx>.json` filename."""
    return int(path.stem[1:])


def load_set(map_name: str) -> list[Layout]:
    """Load `evaluation/layouts/<map>/k<idx>.json` in index order."""
    directory = LAYOUTS / map_name
    paths = sorted(directory.glob("k*.json"), key=_index)
    if not paths:
        raise FileNotFoundError(
            f"no evaluation layouts in {directory}; generate them with `./track {map_name}`"
        )
    layouts: list[Layout] = []
    for path in paths:
        with path.open() as handle:
            specs = json.load(handle)["obstacles"]
        layouts.append(
            tuple(
                Obstacle(
                    s=float(spec["s"]),
                    n=float(spec["n"]),
                    half_length=float(spec["half_length"]),
                    half_width=float(spec["half_width"]),
                )
                for spec in specs
            )
        )
    return layouts


def save_set(map_name: str, layouts: Sequence[Layout], track: Track) -> Path:
    """Write one manifest per layout under `evaluation/layouts/<map>/` and return that directory."""
    directory = LAYOUTS / map_name
    directory.mkdir(parents=True, exist_ok=True)
    for index, layout in enumerate(layouts):
        records = []
        for item in layout:
            x, y, yaw = track.pose(item.s, item.n)
            records.append(
                {
                    "s": item.s,
                    "n": item.n,
                    "half_length": item.half_length,
                    "half_width": item.half_width,
                    "x": x,
                    "y": y,
                    "yaw": yaw,
                }
            )
        manifest = {"map": map_name, "set": map_name, "index": index, "obstacles": records}
        with (directory / f"k{index}.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    return directory
