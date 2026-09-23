"""Choose obstacle zones as station ranges on the racing line, by hand on the map."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from trackgen.paths import TrackPaths
from trackgen.plotting import draw_map

# Dials start on the longest low-curvature stretches of the line.
STRAIGHT_KAPPA_MAX = 0.35
START_EXCLUSION_M = 5.0
MIN_ZONE_M = 1.5

ZoneSpec = int | list[tuple[float, float]]


def parse_zone_spec(text: str) -> ZoneSpec:
    """Parse `--zones`: a zone count, or explicit `a:b,c:d` station ranges in metres."""
    if text.isdigit():
        count = int(text)
        if count < 1:
            raise ValueError("--zones needs at least one zone")
        return count
    ranges: list[tuple[float, float]] = []
    for item in text.split(","):
        a, sep, b = item.partition(":")
        if not sep:
            raise ValueError(f"--zones expects N or a:b,c:d ranges, got {text!r}")
        lo, hi = float(a), float(b)
        if hi <= lo:
            raise ValueError(f"zone {item!r} must have start < end")
        ranges.append((lo, hi))
    return ranges


def _straight_runs(s: np.ndarray, kappa: np.ndarray, lap: float) -> list[tuple[float, float]]:
    """Return low-curvature station runs outside the start exclusion, longest first."""
    ok = np.abs(kappa) <= STRAIGHT_KAPPA_MAX
    ok &= (s >= START_EXCLUSION_M) & (s <= lap - START_EXCLUSION_M)
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if (not flag or i == len(ok) - 1) and start is not None:
            end = i if flag else i - 1
            if s[end] - s[start] >= MIN_ZONE_M:
                runs.append((float(s[start]), float(s[end])))
            start = None
    return sorted(runs, key=lambda r: r[0] - r[1])


def _write(paths: TrackPaths, zones: list[tuple[float, float]]) -> list[list[float]]:
    """Write zones.json and return the ranges rounded to the millimetre."""
    ranges = [[round(a, 3), round(b, 3)] for a, b in sorted(zones)]
    payload = {"valid_zones_s": ranges}
    paths.zones_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {paths.zones_json}: " + ", ".join(f"{a:.2f}-{b:.2f} m" for a, b in ranges))
    return ranges


def _existing(paths: TrackPaths) -> list[tuple[float, float]] | None:
    """Return the ranges already in zones.json, if the file exists."""
    if not paths.zones_json.exists():
        return None
    doc = json.loads(paths.zones_json.read_text())
    return [(float(a), float(b)) for a, b in doc["valid_zones_s"]]


def _dials(paths: TrackPaths, rl: pd.DataFrame, initial: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Open the map with one start/end slider pair per zone; return the ranges on Save."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    free, res, origin = paths.occupancy()
    s = rl["s"].to_numpy(dtype=float)
    lap = float(s[-1])
    x, y = rl["x"].to_numpy(dtype=float), rl["y"].to_numpy(dtype=float)
    xl = x + rl["n_x"].to_numpy() * rl["w_l"].to_numpy()
    yl = y + rl["n_y"].to_numpy() * rl["w_l"].to_numpy()
    xr = x - rl["n_x"].to_numpy() * rl["w_r"].to_numpy()
    yr = y - rl["n_y"].to_numpy() * rl["w_r"].to_numpy()

    n = len(initial)
    fig = plt.figure(figsize=(12, 9 + 0.5 * n))
    ax = fig.add_axes((0.05, 0.12 + 0.05 * n, 0.9, 0.85 - 0.05 * n))
    draw_map(ax, free, res, origin)
    ax.plot(x, y, "b-", lw=1.0, label="raceline")
    ax.plot(xl, yl, "g-", lw=0.6)
    ax.plot(xr, yr, "g-", lw=0.6)
    ticks = np.arange(0.0, lap, 1.0)
    idx = np.searchsorted(s, ticks)
    ax.scatter(x[idx], y[idx], c="k", s=6, zorder=4)
    for t, i in zip(ticks[::2], idx[::2]):
        ax.annotate(f"{t:.0f}", (x[i], y[i]), fontsize=7, textcoords="offset points", xytext=(3, 3))
    ax.scatter([x[0]], [y[0]], c="lime", s=60, zorder=5, label="s = 0")
    ax.legend(loc="upper right")
    ax.set_title("obstacle zones: move the dials, then Save")

    patches: list[Any] = []
    sliders: list[tuple[Slider, Slider]] = []

    def zone_poly(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = min(a, b), max(a, b)
        m = (s >= lo) & (s <= hi)
        return (
            np.concatenate([xl[m], xr[m][::-1]]),
            np.concatenate([yl[m], yr[m][::-1]]),
        )

    def redraw(_val: float | None = None) -> None:
        for patch in patches:
            patch.remove()
        patches.clear()
        for lo_s, hi_s in sliders:
            px, py = zone_poly(lo_s.val, hi_s.val)
            if len(px):
                patches.append(ax.fill(px, py, color="green", alpha=0.35, zorder=3)[0])
        fig.canvas.draw_idle()

    for k, (a, b) in enumerate(initial):
        y0 = 0.04 + 0.05 * (n - 1 - k)
        ax_lo = fig.add_axes((0.12, y0 + 0.022, 0.32, 0.018))
        ax_hi = fig.add_axes((0.55, y0 + 0.022, 0.32, 0.018))
        lo = Slider(ax_lo, f"zone {k + 1} start [m]", 0.0, lap, valinit=a, valstep=0.05)
        hi = Slider(ax_hi, "end [m]", 0.0, lap, valinit=b, valstep=0.05)
        lo.on_changed(redraw)
        hi.on_changed(redraw)
        sliders.append((lo, hi))
    redraw()

    result: list[tuple[float, float]] = []
    ax_save = fig.add_axes((0.88, 0.005, 0.1, 0.03))
    button = Button(ax_save, "Save")

    def on_save(_event: Any) -> None:
        result.extend((min(lo.val, hi.val), max(lo.val, hi.val)) for lo, hi in sliders)
        plt.close(fig)

    button.on_clicked(on_save)
    plt.show(block=True)
    if not result:
        raise SystemExit("zones: window closed without Save")
    return result


def run(paths: TrackPaths, plot: bool, spec: ZoneSpec = 2) -> list[list[float]]:
    """Write zones.json: from the dials when plotting, else from explicit ranges or the existing file."""
    if not paths.raceline_csv.exists():
        raise FileNotFoundError(f"missing raceline: {paths.raceline_csv}")
    rl = pd.read_csv(paths.raceline_csv)
    if isinstance(spec, list):
        return _write(paths, spec)
    if not plot:
        existing = _existing(paths)
        if existing is None:
            raise SystemExit("zones: --no-plot needs --zones a:b,c:d or an existing zones.json")
        print(f"kept {paths.zones_json} ({len(existing)} zones)")
        return _write(paths, existing)
    s = rl["s"].to_numpy(dtype=float)
    lap = float(s[-1])
    initial = _existing(paths) or []
    if len(initial) != spec:
        runs = _straight_runs(s, rl["kappa"].to_numpy(dtype=float), lap)
        initial = sorted(runs[:spec])
        while len(initial) < spec:
            k = len(initial)
            a = START_EXCLUSION_M + k * (lap - 2 * START_EXCLUSION_M) / spec
            initial.append((a, a + MIN_ZONE_M))
    return _write(paths, _dials(paths, rl, initial))
