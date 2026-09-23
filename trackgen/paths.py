"""File locations for one map under tracks/<map>/."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.track import TRACKS, load_occupancy


class TrackPaths:
    """Occupancy inputs in tracks/<map>/ and pipeline outputs in out_dir (default: the same directory)."""

    def __init__(self, map_name: str, out_dir: Path | None = None) -> None:
        self.map_name = map_name
        self.track_dir = TRACKS / map_name
        self.yaml_path = self.track_dir / f"{map_name}.yaml"
        self.out_dir = self.track_dir if out_dir is None else Path(out_dir)
        self.centreline_csv = self.out_dir / "centreline.csv"
        self.raceline_csv = self.out_dir / "raceline.csv"
        self.raceline_mintime_csv = self.out_dir / "raceline_mintime.csv"
        self.zones_json = self.out_dir / "zones.json"
        self.track_yaml = self.out_dir / "track.yaml"

    def occupancy(self) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Free-cell mask (row 0 = image top), resolution and origin, exactly as the simulator reads them."""
        return load_occupancy(self.yaml_path)
