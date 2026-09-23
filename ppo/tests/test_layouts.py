from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ppo.layouts as layouts_module
from ppo.config import ObstaclesConfig
from ppo.layouts import Track, Zones, load_set, sample_layout, save_set

MAP = "lab-20260705"


def _obs() -> ObstaclesConfig:
    return ObstaclesConfig(n_obstacles=2)


def test_sample_layout_bounds() -> None:
    track = Track.load(MAP)
    zones = Zones.load(MAP)
    cfg = _obs()
    half_w = cfg.obstacle_width / 2.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        layout = sample_layout(track, cfg, zones, rng)
        assert len(layout) == 2
        s_values = [item.s for item in layout]
        assert s_values[0] <= s_values[1]
        assert s_values[1] - s_values[0] >= cfg.min_s_distance
        for item in layout:
            assert item.s >= cfg.start_exclusion
            in_zone = any(lo <= item.s <= hi for lo, hi in zones.intervals)
            assert in_zone
            ndx = track.nearest_index(item.s)
            assert item.n >= float(-track.w_r[ndx] + half_w) - 1e-9
            assert item.n <= float(track.w_l[ndx] - half_w) + 1e-9
            if cfg.centre_deadzone > 0.0:
                assert abs(item.n) >= cfg.centre_deadzone
            assert item.half_length == cfg.obstacle_length / 2.0
            assert item.half_width == half_w


def test_load_set_lab_20260906() -> None:
    layouts = load_set("lab-20260906")
    assert len(layouts) == 500
    for layout in layouts:
        assert len(layout) == 2
    first = layouts[0]
    assert first[0].s == 6.849999999999993
    assert first[0].n == 0.3850000000000006
    assert first[0].half_length == 0.175
    assert first[0].half_width == 0.125
    assert first[1].s == 19.150000000000052
    assert first[1].n == 0.9850000000000017
    assert first[1].half_length == 0.175
    assert first[1].half_width == 0.125
    with (layouts_module.LAYOUTS / "lab-20260906" / "k0.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["map"] == "lab-20260906"
    obs = manifest["obstacles"][0]
    assert obs["x"] == 6.926522
    assert obs["y"] == 0.08251450000000055
    assert obs["yaw"] == -0.4084


def test_load_set_missing_names_track_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(layouts_module, "LAYOUTS", tmp_path)
    with pytest.raises(FileNotFoundError, match="./track nowhere"):
        load_set("nowhere")


def test_save_set_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(layouts_module, "LAYOUTS", tmp_path)
    track = Track.load(MAP)
    rng = np.random.default_rng(0)
    written = [sample_layout(track, _obs(), Zones.load(MAP), rng) for _ in range(12)]
    directory = save_set(MAP, written, track)
    assert directory == tmp_path / MAP
    assert load_set(MAP) == written
    with (directory / "k0.json").open() as handle:
        manifest = json.load(handle)
    assert manifest["map"] == MAP
    x, y, yaw = track.pose(written[0][0].s, written[0][0].n)
    assert manifest["obstacles"][0]["x"] == x
    assert manifest["obstacles"][0]["yaw"] == yaw
