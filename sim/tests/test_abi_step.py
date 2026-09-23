"""ABI smoke: reset, step, dtypes, seed reproducibility, and blob load failure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.env import LIBRARY, OBS_SIZE, VecEnv, car_kwargs
from sim.track import blob_path

BLOB = blob_path("lab-20260705")


def _require_lib_and_blob() -> None:
    assert LIBRARY.is_file(), f"missing {LIBRARY}; run `make -C sim`"
    assert BLOB.is_file(), f"missing {BLOB}; run ./setup"


def _venv(num_envs: int, seed: int, sim2real: bool, blob: Path = BLOB) -> VecEnv:
    kwargs = car_kwargs()
    kwargs.update(
        progress_reward=50.0,
        collision_reward=-1.0,
        lap_completion_reward=1.0,
        optimal_lap_time=7.14,
    )
    return VecEnv(num_envs, blob, seed=seed, sim2real=sim2real, **kwargs)


def _rollout(seed: int, steps: int, actions: np.ndarray) -> np.ndarray:
    venv = _venv(2, seed, sim2real=True)
    try:
        obs = venv.reset()
        for _ in range(steps):
            obs, _, _ = venv.step(actions)
        return obs
    finally:
        venv.close()


def test_reset_step_shapes_and_seeds() -> None:
    _require_lib_and_blob()
    venv = _venv(2, 7, sim2real=False)
    try:
        obs = venv.reset()
        assert obs.shape == (2, OBS_SIZE)
        assert obs.dtype == np.float32
        zeros = np.zeros((2, 2), dtype=np.float32)
        obs, reward, terminals = venv.step(zeros)
        assert obs.shape == (2, OBS_SIZE)
        assert np.isfinite(reward).all()
        assert terminals.dtype == np.float32
        ones = np.ones((2, 2), dtype=np.float32)
        obs, reward, terminals = venv.step(ones)
        assert np.isfinite(reward).all()
        assert terminals.dtype == np.float32
    finally:
        venv.close()

    actions = np.zeros((2, 2), dtype=np.float32)
    actions[:, 0] = 0.25
    a = _rollout(123, 50, actions)
    b = _rollout(123, 50, actions)
    c = _rollout(999, 50, actions)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_unloadable_blob_raises(tmp_path: Path) -> None:
    _require_lib_and_blob()
    bad = tmp_path / "bad.blob"
    bad.write_bytes(b"NOPE" + BLOB.read_bytes()[4:])
    with pytest.raises(RuntimeError, match="could not load track blob"):
        _venv(2, 7, sim2real=False, blob=bad)
