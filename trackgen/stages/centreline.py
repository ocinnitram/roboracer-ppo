"""Trace a closed centreline with wall widths from the occupancy map skeleton."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, splev, splprep, splrep
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from trackgen.paths import TrackPaths
from trackgen.plotting import new_axes, show_plot


def splev_xy(u: np.ndarray, tck: object, der: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 2D spline as a pair of float arrays."""
    raw: Any = splev(u, tck, der=der)
    return np.asarray(raw[0], dtype=float), np.asarray(raw[1], dtype=float)


SKELETON_THRESHOLD = 0.05
NON_EDGE = 0.0
TRACK_WIDTH_MARGIN = 0.0
SKELETON_RESOLUTION_M = 0.005
CENTRELINE_RESOLUTION_M = 0.005
SMOOTHING_FACTOR = 5.0


@dataclass(frozen=True)
class _MapMeta:
    resolution: float
    origin: tuple[float, float]


def _load_image(paths: TrackPaths) -> tuple[np.ndarray, _MapMeta]:
    """Load the free-cell mask flipped so row 0 is world-y up."""
    free, resolution, origin = paths.occupancy()
    return np.flipud(free), _MapMeta(resolution, origin)


def _skeleton(free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the skeleton mask and the distance transform sampled on it."""
    dist = np.asarray(distance_transform_edt(free), dtype=float)
    centers = dist > SKELETON_THRESHOLD * dist.max()
    centerline = skeletonize(centers)
    centerline_dist = np.where(centerline, dist, 0)
    return centerline, centerline_dist


def _start_pixel(centerline: np.ndarray, meta: _MapMeta) -> tuple[int, int]:
    """Pick the skeleton pixel closest to the world origin."""
    coords = np.argwhere(centerline)
    world_origin_px = np.array([-meta.origin[0], -meta.origin[1]]) / meta.resolution
    world_origin_yx = np.array([world_origin_px[1], world_origin_px[0]])
    closest = int(np.argmin(np.linalg.norm(coords - world_origin_yx, axis=1)))
    yx = coords[closest]
    return int(yx[1]), int(yx[0])


def _dfs(
    start_x: int,
    start_y: int,
    centerline_dist: np.ndarray,
    reverse: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Walk the skeleton from the start pixel in neighbour order."""
    sys.setrecursionlimit(30000)
    visited: dict[tuple[int, int], bool] = {}
    points: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    directions = [
        (0, -1),
        (-1, 0),
        (0, 1),
        (1, 0),
        (-1, 1),
        (-1, -1),
        (1, 1),
        (1, -1),
    ]

    def walk(point: tuple[int, int]) -> None:
        if point in visited:
            return
        visited[point] = True
        points.append(np.array(point))
        dist = centerline_dist[point[1]][point[0]]
        widths.append(np.array([dist, dist]))
        for dx, dy in directions:
            nxt = (point[0] + dx, point[1] + dy)
            if (
                0 <= nxt[1] < centerline_dist.shape[0]
                and 0 <= nxt[0] < centerline_dist.shape[1]
                and centerline_dist[nxt[1]][nxt[0]] != NON_EDGE
                and nxt not in visited
            ):
                walk(nxt)

    walk((int(start_x), int(start_y)))
    if reverse:
        points = points[::-1]
        widths = widths[::-1]
    return points, widths


def _to_world(
    points: list[np.ndarray],
    widths: list[np.ndarray],
    meta: _MapMeta,
) -> np.ndarray:
    """Convert skeleton pixels and pixel widths into metres."""
    data = np.concatenate((np.array(points), np.array(widths)), axis=1)
    data = data * meta.resolution
    data = data + np.array([meta.origin[0], meta.origin[1], 0.0, 0.0])
    data = data - np.array([0.0, 0.0, TRACK_WIDTH_MARGIN, TRACK_WIDTH_MARGIN])
    return data


def _resample_xyw(data: np.ndarray, spacing: float) -> np.ndarray:
    """Resample x, y, w_r, w_l to even spacing along arc length."""
    x, y, w_r, w_l = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    diffs = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    s = np.concatenate((np.array([0.0]), np.cumsum(diffs)))
    total = float(s[-1])
    n = max(2, int(total / spacing))
    s_new = np.linspace(0.0, total, n)
    fx = interp1d(s, x, kind="cubic")
    fy = interp1d(s, y, kind="cubic")
    fw_r = interp1d(s, w_r, kind="linear")
    fw_l = interp1d(s, w_l, kind="linear")
    return np.column_stack((fx(s_new), fy(s_new), fw_r(s_new), fw_l(s_new)))


def _signed_area(x: np.ndarray, y: np.ndarray) -> float:
    """Return the polygon signed area (negative is clockwise)."""
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _enrich(skeleton_xyw: np.ndarray) -> pd.DataFrame:
    """Fit a smooth periodic spline to the skeleton and return the centreline table."""
    x_raw, y_raw = skeleton_xyw[:, 0], skeleton_xyw[:, 1]
    w_r_raw, w_l_raw = skeleton_xyw[:, 2], skeleton_xyw[:, 3]
    # A periodic fit needs the last point to equal the first exactly.
    if np.hypot(x_raw[0] - x_raw[-1], y_raw[0] - y_raw[-1]) < SKELETON_RESOLUTION_M:
        x_raw, y_raw = x_raw[:-1], y_raw[:-1]
        w_r_raw, w_l_raw = w_r_raw[:-1], w_l_raw[:-1]
    x_raw, y_raw = np.append(x_raw, x_raw[0]), np.append(y_raw, y_raw[0])
    w_r_raw, w_l_raw = np.append(w_r_raw, w_r_raw[0]), np.append(w_l_raw, w_l_raw[0])
    tck_path, u = splprep([x_raw, y_raw], s=SMOOTHING_FACTOR, per=True, k=3)
    u_to_wl = splrep(u, w_l_raw, s=SMOOTHING_FACTOR, per=True, k=3)
    u_to_wr = splrep(u, w_r_raw, s=SMOOTHING_FACTOR, per=True, k=3)
    u_dense = np.linspace(u.min(), u.max(), 10 * len(x_raw))
    x_dense, y_dense = splev_xy(u_dense, tck_path, der=0)
    s_dense = np.concatenate(
        (np.array([0.0]), np.cumsum(np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2)))
    )
    total_length = float(s_dense[-1])
    n_new = max(2, int(total_length / CENTRELINE_RESOLUTION_M))
    s_new = np.linspace(0.0, total_length, n_new)
    s_to_u = interp1d(s_dense, u_dense)
    u_new = np.asarray(s_to_u(s_new), dtype=float)
    x_new, y_new = splev_xy(u_new, tck_path, der=0)
    dx, dy = splev_xy(u_new, tck_path, der=1)
    d2x, d2y = splev_xy(u_new, tck_path, der=2)
    track_yaw = np.arctan2(dy, dx)
    kappa = (dx * d2y - dy * d2x) / ((dx ** 2 + dy ** 2) ** 1.5)
    n_x, n_y = -dy, dx
    mag = np.sqrt(n_x ** 2 + n_y ** 2)
    mag[mag == 0] = 1.0
    n_x = n_x / mag
    n_y = n_y / mag
    w_l = np.asarray(splev(u_new, u_to_wl), dtype=float)
    w_r = np.asarray(splev(u_new, u_to_wr), dtype=float)
    return pd.DataFrame(
        {
            "x": x_new,
            "y": y_new,
            "s": s_new,
            "s_norm": s_new / total_length if total_length > 0 else s_new,
            "track_yaw_rad": track_yaw,
            "kappa": kappa,
            "w_l": w_l,
            "w_r": w_r,
            "n_x": n_x,
            "n_y": n_y,
        }
    )


def run(paths: TrackPaths, direction: str, plot: bool) -> None:
    """Write centreline.csv traced in the given travel direction."""
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    free, meta = _load_image(paths)
    centerline, centerline_dist = _skeleton(free)
    sx, sy = _start_pixel(centerline, meta)
    points, widths = _dfs(sx, sy, centerline_dist, reverse=False)
    world = _to_world(points, widths, meta)
    area = _signed_area(world[:, 0], world[:, 1])
    if ("cw" if area < 0 else "ccw") != direction:
        points, widths = _dfs(sx, sy, centerline_dist, reverse=True)
        world = _to_world(points, widths, meta)
    skeleton = _resample_xyw(world, SKELETON_RESOLUTION_M)
    centre = _enrich(skeleton)
    centre.to_csv(paths.centreline_csv, index=False, float_format="%.4f")
    print(f"wrote {paths.centreline_csv} ({len(centre)} pts, {direction})")
    if plot:
        fig, ax = new_axes((12, 12))
        ax.imshow(free, cmap="gray", origin="lower")
        px = (centre["x"].to_numpy() - meta.origin[0]) / meta.resolution
        py = (centre["y"].to_numpy() - meta.origin[1]) / meta.resolution
        ax.plot(px, py, "b-", lw=1.5)
        ax.scatter([px[0]], [py[0]], c="yellow", s=80, zorder=5)
        step = max(1, len(px) // 200)
        ax.annotate(
            "",
            xy=(px[0] + (px[step] - px[0]) * 8.0, py[0] + (py[step] - py[0]) * 8.0),
            xytext=(px[0], py[0]),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
        )
        ax.set_title(f"centreline {paths.map_name} ({direction})")
        ax.set_aspect("equal")
        show_plot(fig, True)
