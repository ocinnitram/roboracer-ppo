"""A seed with no completions scores pace 0 and lowers the mean score; JSON writes null for NaN."""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import pytest

from evaluation.common import COMPLETED, CRASHED, LayoutResult
from evaluation.report import _mean_block, summarise, write
from ppo.config import load, resolve


def _result(*, outcome: str, lap_times: list[float]) -> LayoutResult:
    return LayoutResult(
        index=0,
        n_obstacles=0,
        outcome=outcome,
        lap_times=lap_times,
        steps=10,
        sim_time=1.0,
        terminal_v=0.0,
    )


def test_uncompleted_seed_lowers_mean_score(tmp_path: Path) -> None:
    cfg = load(resolve("empty"))
    lap = float(cfg.track.opt_time) / 0.93
    done = _result(outcome=COMPLETED, lap_times=[lap, lap])
    fail = _result(outcome=CRASHED, lap_times=[])
    completed = summarise([done], cfg)
    failed = summarise([fail], cfg)
    assert completed["pace"] == pytest.approx(0.93)
    assert completed["success_rate"] == 1.0
    assert failed["success_rate"] == 0.0
    assert failed["collisions"] == 1.0
    assert math.isnan(failed["pace"])
    assert math.isnan(failed["mean_lap_time"])
    assert failed["score"] == 0.0
    mean = _mean_block([completed, failed])
    assert mean["pace"] == pytest.approx(0.93)
    assert mean["score"] == pytest.approx(completed["score"] / 2)
    assert mean["min_success_rate"] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        all_failed = _mean_block([failed, failed])
    assert math.isnan(all_failed["pace"])
    assert all_failed["score"] == 0.0

    path = write(tmp_path, cfg, {}, {1: [done], 2: [fail]})
    text = path.read_text()
    assert "NaN" not in text
    data = json.loads(text)
    assert data["seeds"]["2"]["pace"] is None
    assert data["seeds"]["2"]["mean_lap_time"] is None
    assert data["mean"]["pace"] == pytest.approx(0.93)
    assert data["mean"]["score"] == pytest.approx(completed["score"] / 2)

    with (tmp_path / "layouts.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["seed"], row["outcome"]) for row in rows] == [("1", COMPLETED), ("2", CRASHED)]
