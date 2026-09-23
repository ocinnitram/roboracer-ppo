"""Build an evaluation layout set under evaluation/layouts/<map>/."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from ppo.layouts import LAYOUTS, Layout, Obstacle, Track, Zones, save_set

MIN_S_DISTANCE_M = 5.0
START_EXCLUSION_M = 5.0
N_OBSTACLES = 2
SIGNATURE_DECIMALS = 3
S_STEP_M = 0.05
N_STEP_M = 0.025
CONDITIONING_APPROACH_M = 0.75
CLEARANCE_AFTER_M = 0.5
EGO_LENGTH_M = 0.5
OBSTACLE_WIDTH_M = 0.25
OBSTACLE_LENGTH_M = 0.35

Signature = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PlacementCandidate:
    """One accepted (s, n) placement and its world pose."""

    s: float
    n: float
    x: float
    y: float
    yaw: float

    def as_obstacle(self) -> Obstacle:
        """Turn this candidate into a layout obstacle."""
        return Obstacle(
            s=self.s,
            n=self.n,
            half_length=OBSTACLE_LENGTH_M / 2.0,
            half_width=OBSTACLE_WIDTH_M / 2.0,
        )


def _window_indices(s_track: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Return sample indices whose stations sit in `[start_s, end_s]`."""
    return np.flatnonzero((s_track >= start_s) & (s_track <= end_s))


def enumerate_candidates(
    track: Track, zones: Zones, start_exclusion_m: float
) -> list[PlacementCandidate]:
    """Enumerate the (s, n) grid inside the zones where an obstacle fits the track."""
    s_track = track.s
    s_stop = float(s_track[-1] - CLEARANCE_AFTER_M)
    spans: list[np.ndarray] = []
    for zone_start, zone_stop in zones.intervals:
        lo = max(float(zone_start), start_exclusion_m)
        hi = min(float(zone_stop), s_stop)
        if hi < lo:
            continue
        spans.append(np.arange(lo, hi + 0.5 * S_STEP_M, S_STEP_M))
    s_values = np.concatenate(spans) if spans else np.empty(0)

    candidates: list[PlacementCandidate] = []
    half_span = 0.5 * (OBSTACLE_LENGTH_M + EGO_LENGTH_M)
    for s in s_values:
        window = _window_indices(s_track, s - CONDITIONING_APPROACH_M, s + CLEARANCE_AFTER_M)
        if window.size < 2:
            continue
        centre_index = int(np.argmin(np.abs(s_track - float(s))))
        obstacle_window = _window_indices(s_track, float(s) - half_span, float(s) + half_span)
        if obstacle_window.size == 0:
            continue
        obstacle_n_min = float(np.max(-track.w_r[obstacle_window] + OBSTACLE_WIDTH_M / 2.0))
        obstacle_n_max = float(np.min(track.w_l[obstacle_window] - OBSTACLE_WIDTH_M / 2.0))
        if obstacle_n_min > obstacle_n_max:
            continue
        n_values = np.arange(obstacle_n_min, obstacle_n_max + 0.5 * N_STEP_M, N_STEP_M)
        centre_x = float(track.x[centre_index])
        centre_y = float(track.y[centre_index])
        nx = float(track.n_x[centre_index])
        ny = float(track.n_y[centre_index])
        yaw = float(track.yaw[centre_index])
        for n in n_values:
            n_f = float(n)
            candidates.append(
                PlacementCandidate(
                    s=float(s), n=n_f, x=centre_x + n_f * nx, y=centre_y + n_f * ny, yaw=yaw
                )
            )
    return candidates


def _signature(pairs: Sequence[tuple[float, float]]) -> Signature:
    """Rounded (s, n) sequence used to keep sets disjoint."""
    return tuple(
        (round(s, SIGNATURE_DECIMALS), round(n, SIGNATURE_DECIMALS)) for s, n in pairs
    )


def existing_signatures(root: Path) -> set[Signature]:
    """Collect signatures from every set under `root`."""
    forbidden: set[Signature] = set()
    for manifest_path in root.glob("*/k*.json"):
        with manifest_path.open() as handle:
            specs = json.load(handle).get("obstacles")
        if specs:
            forbidden.add(_signature([(float(spec["s"]), float(spec["n"])) for spec in specs]))
    return forbidden


def build_layouts(
    candidates: list[PlacementCandidate],
    n_layouts: int,
    n_obstacles: int,
    min_s_distance_m: float,
    seed: int,
    forbidden: set[Signature],
) -> list[list[PlacementCandidate]]:
    """Sample stations uniformly, then n uniformly within each station."""
    groups: dict[float, list[PlacementCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(round(candidate.s, SIGNATURE_DECIMALS), []).append(candidate)
    stations = sorted(groups)
    rng = random.Random(seed)
    generated: list[list[PlacementCandidate]] = []
    for layout in range(n_layouts):
        for _ in range(10_000):
            picked: list[PlacementCandidate] = []
            for _ in range(n_obstacles):
                eligible = [
                    s
                    for s in stations
                    if all(abs(s - item.s) >= min_s_distance_m for item in picked)
                ]
                if not eligible:
                    break
                station = rng.choice(eligible)
                picked.append(rng.choice(groups[station]))
            if len(picked) != n_obstacles:
                continue
            picked.sort(key=lambda candidate: candidate.s)
            sig = _signature([(c.s, c.n) for c in picked])
            if sig in forbidden:
                continue
            forbidden.add(sig)
            generated.append(picked)
            break
        else:
            raise RuntimeError(
                f"could not construct layout {layout} with "
                f"{n_obstacles} obstacles at min_s_distance={min_s_distance_m}"
            )
    return generated


def _rectangle_xy(
    x: float, y: float, yaw: float, half_length: float, half_width: float
) -> np.ndarray:
    """World-frame corners of a yaw-aligned rectangle."""
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    local = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return (rot @ local.T).T + np.array([x, y], dtype=float)


def write_contact_sheet(destination: Path, track: Track, layouts: Sequence[Layout]) -> None:
    """Draw every layout's rectangles on the track band into one PNG."""
    n = len(layouts)
    columns = 5 if n <= 50 else 10
    rows = int(np.ceil(n / columns)) if n else 1
    cell = 2.2 if n <= 50 else 1.1
    figure = Figure(figsize=(columns * cell, rows * cell))
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False)
    left_x = track.x + track.w_l * track.n_x
    left_y = track.y + track.w_l * track.n_y
    right_x = track.x - track.w_r * track.n_x
    right_y = track.y - track.w_r * track.n_y
    band_x = np.concatenate([left_x, right_x[::-1]])
    band_y = np.concatenate([left_y, right_y[::-1]])
    for index in range(rows * columns):
        axis = axes[index // columns][index % columns]
        axis.set_aspect("equal")
        axis.axis("off")
        if index >= n:
            continue
        axis.fill(band_x, band_y, color="#eeeeee", zorder=0)
        axis.plot(track.x, track.y, color="#555555", linewidth=0.4)
        for item in layouts[index]:
            x, y, yaw = track.pose(item.s, item.n)
            verts = _rectangle_xy(x, y, yaw, item.half_length, item.half_width)
            axis.add_patch(
                Polygon(verts, closed=True, facecolor="#c44e52", edgecolor="#222222", linewidth=0.4)
            )
        axis.set_title(f"k{index}", fontsize=6, pad=1)
    figure.tight_layout(pad=0.2)
    figure.savefig(destination, dpi=120)


def generate_set(map_name: str, n_layouts: int = 500, seed: int = 500) -> Path:
    """Write evaluation/layouts/<map>/k*.json plus a contact sheet; return the directory."""
    output_dir = LAYOUTS / map_name
    existing = list(output_dir.glob("k*.json"))
    if existing:
        raise FileExistsError(
            f"{output_dir} already has {len(existing)} manifests; evaluation sets are not overwritten"
        )
    track = Track.load(map_name)
    candidates = enumerate_candidates(track, Zones.load(map_name), START_EXCLUSION_M)
    print(
        f"generate map={map_name} layouts={n_layouts} seed={seed} "
        f"candidates={len(candidates)} n_obstacles={N_OBSTACLES} "
        f"min_s_distance={MIN_S_DISTANCE_M} start_exclusion={START_EXCLUSION_M}",
        flush=True,
    )
    picked = build_layouts(
        candidates, n_layouts, N_OBSTACLES, MIN_S_DISTANCE_M, seed, existing_signatures(LAYOUTS)
    )
    layouts = [tuple(candidate.as_obstacle() for candidate in row) for row in picked]
    save_set(map_name, layouts, track)
    # Store the grid candidate pose, not the interpolated track pose.
    for index, row in enumerate(picked):
        path = output_dir / f"k{index}.json"
        with path.open() as handle:
            manifest = json.load(handle)
        for spec, candidate in zip(manifest["obstacles"], row, strict=True):
            spec["x"] = candidate.x
            spec["y"] = candidate.y
            spec["yaw"] = candidate.yaw
        with path.open("w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    sheet = output_dir / f"{map_name}.png"
    write_contact_sheet(sheet, track, layouts)
    print(f"wrote {len(layouts)} layouts to {output_dir} and {sheet.name}", flush=True)
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: python -m evaluation.layouts.generate <map> [--layouts N] [--seed S]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", help="track name under tracks/")
    parser.add_argument("--layouts", type=int, default=500, help="number of layouts (default 500)")
    parser.add_argument("--seed", type=int, default=500, help="sampling seed (default 500)")
    args = parser.parse_args(argv)
    generate_set(args.map, args.layouts, args.seed)


if __name__ == "__main__":
    main()
