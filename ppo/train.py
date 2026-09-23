"""Train a PPO racing policy on the C plant."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, get_args

import numpy as np
import torch as T

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
T.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))

from ppo.agent import PPOAgent
from ppo.config import RUNS, SEED, Config, TrainingConfig, dump, load, resolve, score
from ppo.layouts import Layout, Track, Zones, load_set, sample_layout, to_sim
from ppo.rollout import EVAL_SEED, episode_tick_cap, evaluate, make_venv
from sim.env import COMPLETE, OBS_SIZE

N_PARALLEL_ENV = 32
N_ACTIONS = 2
PROGRESS_FIELDS = ("seed", "step", "time_s", "sps", "score", "success", "pace", "kl", "entropy")


def _set_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch."""
    T.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _is_better_eval(candidate: tuple[float, float], best: tuple[float, float] | None) -> bool:
    """True when candidate beats best under score(), or no best is stored yet."""
    if best is None:
        return True
    return score(candidate[0], candidate[1]) > score(best[0], best[1])


def _decay_parameters(agent: PPOAgent, cfg: Config, step: int) -> None:
    """Decay learning rate, CAPS weights, and entropy bonus linearly over the run."""
    fraction = 1.0 - (step / max(cfg.training.steps, 1))
    lr = cfg.ppo.alpha * fraction
    agent.actor.optimizer.param_groups[0]["lr"] = lr
    agent.critic.optimizer.param_groups[0]["lr"] = lr
    agent.set_caps_scale(fraction)
    agent.set_c2(cfg.ppo.c2_i * fraction + cfg.ppo.c2_f * (1.0 - fraction))


def _eval_layouts(cfg: Config) -> list[Layout]:
    """Evaluation layouts: the map's held-out set with obstacles, else one empty layout."""
    if cfg.obstacles.n_obstacles == 0:
        return [()]
    return load_set(cfg.track.map)


def _assign_layout(
    venv: Any, cfg: Config, track: Track, zones: Zones, i: int, rng: np.random.Generator
) -> None:
    """Reset env i onto a freshly sampled layout."""
    layout = sample_layout(track, cfg.obstacles, zones, rng)
    venv.reset_env(i)
    venv.set_obstacles(i, to_sim(layout))


def _redraw_finished_layouts(
    venv: Any,
    cfg: Config,
    track: Track,
    zones: Zones,
    rng: np.random.Generator,
    dones: Sequence[bool],
    states: np.ndarray,
) -> None:
    """Give finished envs a new layout per training.layout_switch, updating states in place."""
    every_episode = cfg.training.layout_switch == "episode"
    for i, done in enumerate(dones):
        if done and (every_episode or venv.done_kind(i) == COMPLETE):
            _assign_layout(venv, cfg, track, zones, i, rng)
            states[i] = venv.obs[i]


def _collect_step(states: np.ndarray, agent: PPOAgent, venv: Any) -> tuple[np.ndarray, list[bool]]:
    """Sample actions, step the plant, and store the interleaved transitions."""
    actions, log_probs, values = agent.choose_action_batched(states)
    next_states, rewards, terminals = venv.step(actions)
    dones = [bool(flag) for flag in terminals]
    for i in range(N_PARALLEL_ENV):
        agent.store_transition(
            states[i], actions[i], log_probs[i], float(values[i].item()), float(rewards[i]), dones[i]
        )
    return next_states, dones


def _run_dir(cfg: Config, seed: int) -> Path:
    """Create runs/<name>/, refuse existing checkpoints for this seed, and write config.yaml."""
    run_dir = RUNS / cfg.training.run_name
    prefixes = (f"_actor_seed_{seed}", f"_critic_seed_{seed}")
    if run_dir.is_dir():
        existing = sorted(
            path.name
            for path in run_dir.iterdir()
            if any(path.name == p or path.name.startswith(p + "_") for p in prefixes)
        )
        if existing:
            raise SystemExit(
                f"run {cfg.training.run_name!r} already has checkpoints for seed {seed}: "
                f"{', '.join(existing)}; pick another --name or --seed"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(dump(cfg))
    return run_dir


def make_agent(cfg: Config, seed: int, run_dir: Path) -> PPOAgent:
    """Construct the actor-critic for this seed with its files under run_dir."""
    return PPOAgent(
        n_actions=N_ACTIONS,
        input_dims=OBS_SIZE,
        gamma=cfg.ppo.gamma,
        alpha=cfg.ppo.alpha,
        gae_lambda=cfg.ppo.gae_lambda,
        policy_clip=cfg.ppo.clip,
        batch_size=cfg.ppo.batch_size,
        n_epochs=cfg.ppo.n_epochs,
        log_std_init=cfg.ppo.log_std_i,
        initial_c2=cfg.ppo.c2_i,
        normalise_advantages=cfg.ppo.normalise_advantages,
        target_kl=cfg.ppo.target_kl,
        c1=cfg.ppo.c1,
        base_path=run_dir.parent,
        run_name=run_dir.name,
        seed=seed,
        hidden_size=cfg.ppo.hidden_size,
        caps_lambda_t=cfg.ppo.caps_lambda_t,
        caps_lambda_s=cfg.ppo.caps_lambda_s,
        caps_sigma_s=cfg.ppo.caps_sigma_s,
        caps_lambda_t_v=cfg.ppo.caps_lambda_t_v,
        caps_lambda_s_v=cfg.ppo.caps_lambda_s_v,
    )


def run(cfg: Config, seed: int, use_wandb: bool = False) -> None:
    """Train one seed, saving checkpoints and progress.csv under runs/<name>/."""
    run_dir = _run_dir(cfg, seed)
    # wandb.init may draw from the global RNGs, so it runs before seeding.
    wandb_run = _init_wandb(cfg, seed, run_dir) if use_wandb else None
    _set_seeds(seed)
    agent = make_agent(cfg, seed, run_dir)

    total_steps = cfg.training.steps
    eval_interval = cfg.training.eval_interval
    ckpt_interval = cfg.training.ckpt_interval
    track = Track.load(cfg.track.map)
    zones = Zones.load(cfg.track.map) if cfg.obstacles.n_obstacles > 0 else None
    rng = np.random.default_rng(seed)
    eval_layouts = _eval_layouts(cfg)

    venv = make_venv(cfg, N_PARALLEL_ENV, seed)
    lap_cap = int(cfg.training.num_laps) + 1
    tick_cap = episode_tick_cap(cfg)
    for i in range(N_PARALLEL_ENV):
        venv.set_lap_cap(i, lap_cap)
        venv.set_tick_cap(i, tick_cap)
    states = venv.reset()
    if zones is not None:
        for i in range(N_PARALLEL_ENV):
            _assign_layout(venv, cfg, track, zones, i, rng)
        states = venv.obs.copy()

    progress_path = run_dir / "progress.csv"
    new_file = not progress_path.exists()
    step = 0
    start_time = time.time()
    last_update_time, last_update_step = start_time, 0
    next_eval_step = eval_interval
    next_ckpt_step = ckpt_interval
    best_eval: tuple[float, float] | None = None
    evals_since_improve = 0
    stop_reason: str | None = None

    print(f"Training seed {seed} for {total_steps} steps on {N_PARALLEL_ENV} envs")
    try:
        with progress_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROGRESS_FIELDS)
            if new_file:
                writer.writeheader()
            while step < total_steps:
                _decay_parameters(agent, cfg, step)
                next_states, dones = _collect_step(states, agent, venv)
                step += N_PARALLEL_ENV

                if len(agent.rollout_buffer.rewards) >= cfg.ppo.rollout:
                    agent.store_rollout_buffer()
                    diag = agent.learn(next_states, dones, N_PARALLEL_ENV)
                    if not (np.isfinite(diag["actor_loss"]) and np.isfinite(diag["critic_loss"])):
                        stop_reason = "non-finite loss"
                        break
                    now = time.time()
                    sps = (step - last_update_step) / max(now - last_update_time, 1e-9)
                    last_update_time, last_update_step = now, step
                    row: dict[str, Any] = {
                        "seed": seed,
                        "step": step,
                        "time_s": f"{now - start_time:.1f}",
                        "sps": f"{sps:.0f}",
                        "score": "",
                        "success": "",
                        "pace": "",
                        "kl": f"{diag['approx_kl']:.5f}",
                        "entropy": f"{diag['entropy']:.4f}",
                    }
                    if step >= next_eval_step:
                        next_eval_step = (step // eval_interval + 1) * eval_interval
                        metrics = evaluate(agent, cfg, eval_layouts, EVAL_SEED)
                        result = (metrics["success_rate"], metrics["pace"])
                        row["score"] = f"{score(*result):.4f}"
                        row["success"] = f"{result[0]:.4f}"
                        row["pace"] = f"{result[1]:.4f}"
                        if _is_better_eval(result, best_eval):
                            best_eval = result
                            evals_since_improve = 0
                            agent.save_models_with_suffix("_best_eval")
                        else:
                            evals_since_improve += 1
                    writer.writerow(row)
                    handle.flush()
                    if wandb_run is not None:
                        wandb_run.log(
                            {key: float(value) for key, value in row.items() if value != "" and key != "seed"},
                            step=step,
                        )
                    print(" | ".join(f"{key} {value}" for key, value in row.items() if value != ""))
                    if evals_since_improve >= cfg.training.patience_evals:
                        stop_reason = f"no eval improvement for {cfg.training.patience_evals} evals"
                        break

                if zones is not None:
                    _redraw_finished_layouts(venv, cfg, track, zones, rng, dones, next_states)
                states = next_states

                if step >= next_ckpt_step:
                    agent.save_models_with_suffix(f"_ckpt_{step}")
                    next_ckpt_step += ckpt_interval

        if best_eval is None:
            metrics = evaluate(agent, cfg, eval_layouts, EVAL_SEED)
            agent.save_models_with_suffix("_best_eval")
            print(f"eval at end | success {metrics['success_rate']:.4f} | pace {metrics['pace']:.4f}")
        if stop_reason:
            print(f"Stopped early at {step} steps: {stop_reason}")
        agent.save_models()
    finally:
        venv.close()
        if wandb_run is not None:
            wandb_run.finish()


def _check_wandb_login() -> None:
    """Exit before any work when wandb would fail later; offline mode needs no account."""
    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return
    import wandb

    try:
        wandb.login(verify=True)
    except Exception as exc:
        raise SystemExit(f"no wandb login; run `uv run wandb login` or set WANDB_API_KEY ({exc})") from exc


def _init_wandb(cfg: Config, seed: int, run_dir: Path) -> Any:
    """Start a wandb run grouped by run name, one job type per seed."""
    import wandb

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "roboracer-ppo"),
        group=cfg.training.run_name,
        job_type=f"seed_{seed}",
        name=f"{cfg.training.run_name}-seed{seed}",
        config={**cfg.model_dump(), "seed": seed},
        dir=str(run_dir),
    )


def _run_name(cfg: Config, args: argparse.Namespace) -> str:
    """Default run name: the config's, plus -<map> and -<layout_switch> when the flags change them."""
    name = cfg.training.run_name
    if args.track and args.track != cfg.track.map:
        name += f"-{args.track}"
    if args.layout_switch and args.layout_switch != cfg.training.layout_switch:
        name += f"-{args.layout_switch}"
    return name


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for ./train."""
    parser = argparse.ArgumentParser(prog="./train", description="Train a PPO racing policy.")
    parser.add_argument(
        "config",
        nargs="?",
        default="empty",
        help="empty, obstacles, or a config.yaml path (default empty)",
    )
    parser.add_argument("--track", metavar="MAP", help="train on tracks/MAP instead of the config's map")
    parser.add_argument("--seed", type=int, default=SEED, metavar="N", help=f"random seed (default {SEED})")
    parser.add_argument("--steps", type=int, metavar="N", help="total environment steps (default from the config)")
    parser.add_argument("--name", help="run name under runs/ (default derived from the config)")
    parser.add_argument(
        "--layout-switch",
        choices=get_args(TrainingConfig.model_fields["layout_switch"].annotation),
        help="when an obstacle env gets a new layout: every episode, or after num_laps complete",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="also log progress to Weights & Biases (project roboracer-ppo, or $WANDB_PROJECT)",
    )
    args = parser.parse_args(argv)
    if args.wandb:
        _check_wandb_login()

    path = resolve(args.config)
    overrides: dict[str, Any] = {"training.run_name": args.name or _run_name(load(path), args)}
    if args.track:
        overrides["track.map"] = args.track
    if args.layout_switch:
        overrides["training.layout_switch"] = args.layout_switch
    if args.steps is not None:
        overrides["training.steps"] = args.steps
    cfg = load(path, overrides)
    print(dump(cfg), flush=True)
    run(cfg, args.seed, use_wandb=args.wandb)


if __name__ == "__main__":
    main(sys.argv[1:])
