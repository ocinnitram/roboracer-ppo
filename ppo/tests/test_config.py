from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import ppo.config as config
import ppo.train as train
from ppo.config import Config, dump, load, resolve, score
from sim.env import COLLISION, COMPLETE
from sim.track import TRACKS

EMPTY = config.CONFIGS / "empty.yaml"


def test_resolve_name() -> None:
    assert resolve("empty") == EMPTY


def test_resolve_path() -> None:
    assert resolve(str(EMPTY)) == EMPTY.resolve()


def test_resolve_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RUNS", tmp_path)
    run_cfg = tmp_path / "some-run" / "config.yaml"
    run_cfg.parent.mkdir()
    run_cfg.write_text("run: 1\n")
    assert resolve("some-run") == run_cfg


def test_resolve_missing() -> None:
    with pytest.raises(FileNotFoundError):
        resolve("no-such-config")


def test_override_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        load(EMPTY, {"ppo.not_a_real_key": 1.0})


def test_track_yaml_merge() -> None:
    cfg = load(EMPTY)
    with (TRACKS / cfg.track.map / "track.yaml").open() as handle:
        track = yaml.safe_load(handle)
    assert cfg.track.opt_time == pytest.approx(float(track["opt_time"]))
    assert cfg.track.max_speed_mps == track.get("max_speed_mps")


def test_map_override_switches_track(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TRACKS", tmp_path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "track.yaml").write_text("opt_time: 9.5\nmax_speed_mps: 3.0\n")
    cfg = load(EMPTY, {"track.map": "other", "ppo.alpha": 0.001})
    assert cfg.track.map == "other"
    assert cfg.track.opt_time == pytest.approx(9.5)
    assert cfg.track.max_speed_mps == pytest.approx(3.0)
    assert cfg.ppo.alpha == pytest.approx(0.001)


def test_best_eval_follows_score_not_lexicographic() -> None:
    """A slight success drop with a larger pace gain wins under score()."""
    lex_winner = (0.80, 0.90)
    score_winner = (0.79, 1.05)
    assert lex_winner > score_winner
    assert score(*score_winner) > score(*lex_winner)
    assert train._is_better_eval(score_winner, lex_winner)
    assert not train._is_better_eval(lex_winner, score_winner)
    assert train._is_better_eval(lex_winner, None)


@pytest.mark.parametrize(("layout_switch", "redrawn"), [("complete", {1}), ("episode", {0, 1})])
def test_layout_switch_rule(
    layout_switch: str, redrawn: set[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env 0 crashed, env 1 completed, env 2 still running."""
    cfg = load(EMPTY, {"training.layout_switch": layout_switch})
    assigned: list[int] = []
    monkeypatch.setattr(
        train, "_assign_layout", lambda venv, cfg, track, zones, i, rng: assigned.append(i)
    )
    kinds = {0: COLLISION, 1: COMPLETE}

    class Venv:
        obs = np.ones((3, 1))

        def done_kind(self, i: int) -> int:
            return kinds[i]

    states = np.zeros((3, 1))
    train._redraw_finished_layouts(Venv(), cfg, None, None, None, [True, True, False], states)
    assert set(assigned) == redrawn
    assert [bool(states[i, 0]) for i in range(3)] == [i in redrawn for i in range(3)]


def test_layout_switch_rejects_unknown() -> None:
    with pytest.raises(Exception):
        load(EMPTY, {"training.layout_switch": "never"})


def test_dump_round_trip() -> None:
    cfg = load(EMPTY, {"training.steps": 12345})
    again = Config(**yaml.safe_load(dump(cfg)))
    assert again.model_dump() == cfg.model_dump()
    assert again.training.steps == 12345
