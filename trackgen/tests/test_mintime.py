"""Run the minimum-time stage on the golden lab-20260705 raceline and check the output table."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from trackgen.paths import TrackPaths
from trackgen.stages import mintime

MAP = "lab-20260705"
RACELINE = Path(__file__).resolve().parent / "golden" / MAP / "raceline.csv"
# Lap time on RACELINE at the two-decimal precision of opt_time; the tolerance is half the last digit.
LAP_S = 7.04
LAP_TOL_S = 0.005
COLUMNS = ("x", "y", "s", "s_norm", "track_yaw_rad", "kappa", "w_l", "w_r", "n_x", "n_y", "v_ref")


def _lap_from_csv(df: pd.DataFrame) -> float:
    """Integrate ds/v_ref around the closed mintime raceline."""
    xy = df[["x", "y"]].to_numpy(dtype=float)
    closed = np.vstack([xy, xy[:1]])
    ds = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))
    v = np.maximum(df["v_ref"].to_numpy(dtype=float), 1e-3)
    return float(np.sum(ds / v))


def test_mintime_lap_time(tmp_path: Path) -> None:
    """Require a v_ref table, the recorded lap time, and a lap consistent with the written profile."""
    paths = TrackPaths(MAP, out_dir=tmp_path)
    shutil.copy(RACELINE, paths.raceline_csv)
    lap = mintime.run(paths, plot=False)
    assert abs(lap - LAP_S) <= LAP_TOL_S, f"mintime lap {lap:.4f}s, expected {LAP_S}s"
    df = pd.read_csv(paths.raceline_mintime_csv)
    assert list(df.columns) == list(COLUMNS)
    assert len(df) > 100
    csv_lap = _lap_from_csv(df)
    assert abs(csv_lap - lap) <= 0.02, f"returned {lap:.4f}s vs csv {csv_lap:.4f}s"
