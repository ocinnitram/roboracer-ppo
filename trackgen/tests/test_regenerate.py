"""Regenerate the lab-20260705 raceline from its occupancy map and compare it with the golden output."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trackgen.paths import TrackPaths
from trackgen.stages import centreline, raceline

MAP = "lab-20260705"
GOLDEN_RACELINE = Path(__file__).resolve().parent / "golden" / MAP / "raceline.csv"


def test_raceline_parity(tmp_path: Path) -> None:
    """Centreline then raceline into tmp; require last-digit agreement with the golden file."""
    paths = TrackPaths(MAP, out_dir=tmp_path)
    centreline.run(paths, "cw", plot=False)
    raceline.run(paths, "cw", plot=False)
    expected = pd.read_csv(GOLDEN_RACELINE)
    got = pd.read_csv(paths.raceline_csv)
    assert list(got.columns) == list(expected.columns)
    assert len(got) == len(expected)
    for col in expected.columns:
        delta = np.abs(got[col].to_numpy(dtype=float) - expected[col].to_numpy(dtype=float))
        assert float(delta.max()) <= 0.0001 + 1e-12, f"{col} max abs {delta.max()}"
