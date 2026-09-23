"""Evaluate a trained run on the empty track or its map's obstacle layouts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from evaluation import policy, render, report
from evaluation.common import LayoutResult
from ppo.agent import PPOAgent
from ppo.config import Config, load
from ppo.layouts import Layout, load_set
from ppo.train import RUNS, make_agent


def _checkpoint_suffix(checkpoint: str) -> str:
    """Map a checkpoint name to the actor-file suffix."""
    if checkpoint == "best":
        return "_best_eval"
    if checkpoint == "final":
        return ""
    if checkpoint.isdigit():
        return f"_ckpt_{checkpoint}"
    raise ValueError(f"checkpoint must be best, final, or a step number, got {checkpoint!r}")


def _discover_seeds(run_dir: Path, suffix: str) -> list[int]:
    """Seeds that have an actor file with exactly this checkpoint suffix."""
    prefix = "_actor_seed_"
    seeds: list[int] = []
    for path in run_dir.glob(f"{prefix}*{suffix}"):
        rest = path.name[len(prefix) :]
        if suffix:
            rest = rest[: -len(suffix)]
        if rest.isdigit():
            seeds.append(int(rest))
    return sorted(seeds)


def load_agent(run_dir: Path, cfg: Config, seed: int, suffix: str) -> PPOAgent:
    """Build the agent for one seed and load its actor weights."""
    path = run_dir / f"_actor_seed_{seed}{suffix}"
    if not path.is_file():
        raise SystemExit(f"no actor file {path}")
    agent = make_agent(cfg, seed, run_dir)
    agent.actor.path = path
    agent.actor.load_checkpoint()
    agent.actor.eval()
    return agent


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the eval command line."""
    parser = argparse.ArgumentParser(prog="eval", description=__doc__)
    parser.add_argument(
        "run",
        nargs="?",
        default="empty",
        help="run name under runs/, or a run directory such as pretrained/empty (default empty)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="one seed (default: every seed in the run)"
    )
    parser.add_argument(
        "--checkpoint", default="best", help="best, final, or a training step (default best)"
    )
    track = parser.add_mutually_exclusive_group()
    track.add_argument(
        "--empty", action="store_true", help="evaluate on the empty track even if trained with obstacles"
    )
    track.add_argument(
        "--obstacles", action="store_true", help="evaluate on the obstacle layouts even if trained without"
    )
    parser.add_argument(
        "--render", action="store_true", help="watch one seed drive the layouts live instead of scoring"
    )
    parser.add_argument(
        "--layout", type=int, default=None, help="first obstacle layout to render (implies --obstacles)"
    )
    args = parser.parse_args(argv)
    if args.layout is not None and not args.render:
        parser.error("--layout only applies to --render")
    if args.layout is not None and args.empty:
        parser.error("--layout needs the obstacle layouts; drop --empty")
    try:
        _checkpoint_suffix(args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ./eval [run] [--seed N] [--checkpoint best|final|<step>] [--empty|--obstacles] [--render]."""
    args = _parse_args(argv)
    run_dir = RUNS / args.run
    if not run_dir.is_dir() and Path(args.run).is_dir():
        run_dir = Path(args.run)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise SystemExit(f"no trained run at {run_dir}")
    cfg = load(config_path)
    suffix = _checkpoint_suffix(args.checkpoint)
    seeds = [args.seed] if args.seed is not None else _discover_seeds(run_dir, suffix)
    if not seeds:
        raise SystemExit(f"no _actor_seed_*{suffix} files in {run_dir}")
    obstacles = args.obstacles or args.layout is not None
    empty = args.empty or (cfg.obstacles.n_obstacles == 0 and not obstacles)
    layouts: list[Layout] = [()] if empty else load_set(cfg.track.map)
    kind = "empty" if empty else "obstacles"
    sim2real = bool(cfg.training.sim2real)
    num_laps = int(cfg.training.num_laps)
    if args.render:
        first = args.layout or 0
        if not 0 <= first < len(layouts):
            raise SystemExit(f"--layout must be in [0, {len(layouts)})")
        agent = load_agent(run_dir, cfg, seeds[0], suffix)
        print(f"render: run={args.run} seed={seeds[0]} layouts={kind} from {first}", flush=True)
        render.run(agent, cfg, layouts, first=first, sim2real=sim2real, num_laps=num_laps)
        return
    out_dir = run_dir / "eval" / f"{datetime.now():%Y%m%d-%H%M%S}_{kind}"
    print(
        f"eval: run={args.run} map={cfg.track.map} checkpoint={args.checkpoint} "
        f"seeds={','.join(map(str, seeds))} layouts={kind}:{len(layouts)} "
        f"laps={num_laps} sim2real={sim2real} out={out_dir}",
        flush=True,
    )
    per_seed: dict[int, list[LayoutResult]] = {}
    for seed in seeds:
        agent = load_agent(run_dir, cfg, seed, suffix)
        per_seed[seed] = policy.run(agent, cfg, layouts, sim2real=sim2real, num_laps=num_laps)
    eval_args = {
        "run": args.run,
        "checkpoint": args.checkpoint,
        "seeds": seeds,
        "layouts": kind,
        "laps": num_laps,
        "sim2real": sim2real,
    }
    summary_path = report.write(out_dir, cfg, eval_args, per_seed)
    report.print_summary(per_seed, cfg)
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
