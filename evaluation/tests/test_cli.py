"""CLI defaults, checkpoint suffix, and seed discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.cli import _checkpoint_suffix, _discover_seeds, _parse_args


def test_parse_defaults() -> None:
    args = _parse_args([])
    assert args.run == "empty"
    assert args.seed is None
    assert args.checkpoint == "best"
    assert args.empty is False


def test_parse_rejects_unknown_checkpoint() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["some-run", "--checkpoint", "best_eval"])


def test_checkpoint_suffix_mapping() -> None:
    assert _checkpoint_suffix("best") == "_best_eval"
    assert _checkpoint_suffix("final") == ""
    assert _checkpoint_suffix("1000000") == "_ckpt_1000000"
    with pytest.raises(ValueError):
        _checkpoint_suffix("latest")


def test_discover_seeds_from_run_dir(tmp_path: Path) -> None:
    (tmp_path / "_actor_seed_17_best_eval").write_text("x")
    (tmp_path / "_actor_seed_76_best_eval").write_text("x")
    (tmp_path / "_actor_seed_17").write_text("x")
    (tmp_path / "_actor_seed_122_ckpt_1000").write_text("x")
    (tmp_path / "_critic_seed_5_best_eval").write_text("x")
    assert _discover_seeds(tmp_path, _checkpoint_suffix("best")) == [17, 76]
    assert _discover_seeds(tmp_path, _checkpoint_suffix("final")) == [17]
    assert _discover_seeds(tmp_path, _checkpoint_suffix("1000")) == [122]
