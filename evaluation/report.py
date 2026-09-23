"""JSON summary, per-layout TSV, and printed table."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.common import CRASHED, LayoutResult
from ppo.config import Config, score

_NUMERIC = ("success_rate", "pace", "mean_lap_time", "collisions", "n_layouts", "score")
_TABLE = (
    ("seed", "seed"), ("success", "success_rate"), ("pace", "pace"),
    ("lap", "mean_lap_time"), ("coll", "collisions"), ("layouts", "n_layouts"),
    ("score", "score"),
)
_TSV = ("seed", "layout", "n_obstacles", "outcome", "laps_s", "term_v", "steps")


def summarise(results: Sequence[LayoutResult], cfg: Config) -> dict:
    """Aggregate one seed's layouts into the printed and JSON metrics."""
    n = len(results)
    means = [float(np.mean(r.lap_times)) for r in results if r.success and r.lap_times]
    mean_lap = float(np.mean(means)) if means else float("nan")
    pace = float(cfg.track.opt_time) / mean_lap if means else float("nan")
    success_rate = sum(1 for r in results if r.success) / n if n else 0.0
    return {
        "success_rate": success_rate,
        "pace": pace,
        "mean_lap_time": mean_lap,
        "collisions": float(sum(1 for r in results if r.outcome == CRASHED)),
        "n_layouts": float(n),
        "score": score(success_rate, pace if means else 0.0),
    }


def _mean_block(summaries: Sequence[dict]) -> dict[str, Any]:
    """Mean of the finite values of each numeric metric over seeds, plus min_success_rate."""
    out: dict[str, Any] = {}
    for key in _NUMERIC:
        values = [float(row[key]) for row in summaries if math.isfinite(row[key])]
        out[key] = sum(values) / len(values) if values else float("nan")
    rates = [float(row["success_rate"]) for row in summaries]
    out["min_success_rate"] = min(rates) if rates else 0.0
    return out


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None so dumps writes null."""
    if isinstance(obj, dict):
        return {key: _json_safe(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(val) for val in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write(
    out_dir: Path, cfg: Config, args: dict, per_seed: dict[int, list[LayoutResult]]
) -> Path:
    """Write summary.json and layouts.tsv; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_block: dict[str, Any] = {}
    summaries: list[dict] = []
    tsv_rows: list[list[object]] = []
    for seed in sorted(per_seed):
        results = per_seed[seed]
        summary = summarise(results, cfg)
        summaries.append(summary)
        seed_block[str(seed)] = {**summary, "layouts": [r.to_dict() for r in results]}
        for r in results:
            tsv_rows.append([
                seed, r.index, r.n_obstacles, r.outcome,
                ",".join(f"{t:.4f}" for t in r.lap_times), r.terminal_v, r.steps,
            ])
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(_json_safe({
        "config": cfg.model_dump(), "eval": args, "seeds": seed_block,
        "mean": _mean_block(summaries),
    }), indent=2, default=str, allow_nan=False) + "\n")
    with (out_dir / "layouts.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(_TSV)
        writer.writerows(tsv_rows)
    return json_path


def print_summary(per_seed: dict[int, list[LayoutResult]], cfg: Config) -> None:
    """Print an aligned per-seed table plus a mean row."""
    summaries = [summarise(per_seed[seed], cfg) for seed in sorted(per_seed)]
    rows = [{"seed": str(seed), **s} for seed, s in zip(sorted(per_seed), summaries)]
    if summaries:
        rows.append({"seed": "mean", **_mean_block(summaries)})
    cells = [[title for title, _ in _TABLE]]
    for row in rows:
        line: list[str] = []
        for _, key in _TABLE:
            value = row[key]
            if isinstance(value, float):
                line.append(f"{value:.4f}" if math.isfinite(value) else "")
            else:
                line.append(str(value))
        cells.append(line)
    widths = [max(len(row[i]) for row in cells) for i in range(len(_TABLE))]
    print("\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in cells
    ), flush=True)
