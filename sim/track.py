"""Compile tracks/<map>/ (occupancy map + raceline.csv) into the simulator's track blob."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

TRACKS = Path(__file__).resolve().parents[1] / "tracks"

MAGIC = struct.unpack("<I", b"RRBC")[0]
VERSION = 1
HEADER_BYTES = 64
# magic, version, W, H, resolution, origin_x, origin_y, n_rline, edt_offset, rline_offset; zero-padded to 64 bytes.
HEADER_FMT = "<2I 2i f 2f i 2i"

RLINE_COLS = [
    "x", "y", "s", "s_norm", "track_yaw_rad",
    "kappa", "w_l", "w_r", "n_x", "n_y",
]


def blob_path(map_name: str) -> Path:
    """Path of the compiled blob for a map."""
    return TRACKS / map_name / f"{map_name}.blob"


def _image_path(yaml_path: Path, cfg: dict) -> Path:
    """Occupancy image named by a map yaml, relative to it."""
    image_path = yaml_path.parent / cfg["image"]
    if not image_path.is_file():
        raise FileNotFoundError(f"missing {image_path}")
    return image_path


def _sources(map_name: str) -> tuple[Path, Path, Path]:
    """Map yaml, the occupancy image it names, and raceline.csv."""
    track_dir = TRACKS / map_name
    yaml_path = track_dir / f"{map_name}.yaml"
    csv_path = track_dir / "raceline.csv"
    for path in (yaml_path, csv_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
    return yaml_path, _image_path(yaml_path, yaml.safe_load(yaml_path.read_text())), csv_path


def load_occupancy(yaml_path: Path) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Boolean free mask in image orientation (row 0 = top), resolution [m/cell], and origin (x, y) [m]."""
    cfg = yaml.safe_load(Path(yaml_path).read_text())
    img = np.array(Image.open(_image_path(Path(yaml_path), cfg)).convert("L"), dtype=np.float64)
    p = img / 255.0 if int(cfg.get("negate", 0)) else (255.0 - img) / 255.0
    # Only confidently free cells are free; unknown cells between the thresholds count as walls.
    free = p < float(cfg["free_thresh"])
    origin = cfg["origin"]
    return free, float(cfg["resolution"]), (float(origin[0]), float(origin[1]))


def build_signed_edt(free_mask: np.ndarray, resolution: float) -> np.ndarray:
    """Signed distance to the nearest wall in metres (positive in free space), float32."""
    dist_to_wall = np.asarray(distance_transform_edt(free_mask), dtype=np.float64)
    dist_to_free = np.asarray(distance_transform_edt(1 - free_mask), dtype=np.float64)
    return ((dist_to_wall - dist_to_free) * resolution).astype(np.float32)


def load_raceline(csv_path: Path) -> np.ndarray:
    """Load the 10-column raceline, checking its header."""
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    if header != RLINE_COLS:
        raise ValueError(f"raceline header mismatch in {csv_path}: {header}")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(RLINE_COLS):
        raise ValueError(f"raceline shape {data.shape}, expected (N, {len(RLINE_COLS)})")
    return data


def pack_blob(out_path: Path, edt: np.ndarray, rline: np.ndarray, resolution: float, origin) -> None:
    """Write the header, the row-major EDT, and the raceline into one blob."""
    H, W = edt.shape
    edt_offset = HEADER_BYTES
    rline_offset = edt_offset + edt.size * 4
    header = struct.pack(
        HEADER_FMT,
        MAGIC,
        VERSION,
        W,
        H,
        float(resolution),
        float(origin[0]),
        float(origin[1]),
        rline.shape[0],
        edt_offset,
        rline_offset,
    )
    header += b"\x00" * (HEADER_BYTES - len(header))
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(edt, dtype="<f4").tobytes())
        f.write(np.ascontiguousarray(rline, dtype="<f4").tobytes())


def compile_map(map_name: str) -> Path:
    """Write tracks/<map>/<map>.blob."""
    yaml_path, _, csv_path = _sources(map_name)
    free, resolution, origin = load_occupancy(yaml_path)
    # The blob grid is world-y-up: row 0 is the bottom of the image.
    edt = build_signed_edt(np.flipud(free).astype(np.uint8), resolution)
    rline = load_raceline(csv_path)

    out_blob = blob_path(map_name)
    pack_blob(out_blob, edt, rline, resolution, origin)
    print(f"{map_name}: {edt.shape[1]}x{edt.shape[0]} cells, {rline.shape[0]} raceline pts -> {out_blob}")
    return out_blob


def ensure_blob(map_name: str) -> Path:
    """Compile the blob when it is missing or older than any of its sources."""
    path = blob_path(map_name)
    yaml_path, image_path, csv_path = _sources(map_name)
    if path.is_file() and all(path.stat().st_mtime >= src.stat().st_mtime for src in (yaml_path, image_path, csv_path)):
        return path
    return compile_map(map_name)


def main() -> None:
    """Compile the named map."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map", help="track name under tracks/")
    args = ap.parse_args()
    compile_map(args.map)


if __name__ == "__main__":
    main()
