"""Batched mean-action evaluation over a list of layouts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch as T

from ppo.agent import PPOAgent
from ppo.config import Config
from ppo.layouts import Layout, to_sim
from sim.env import COLLISION, COMPLETE, VecEnv, car_kwargs
from sim.track import ensure_blob

EVAL_SEED = 1
EVAL_TIMEOUT_FACTOR = 6.0
# 45 s of plant time per 7.02 s of minimum-time lap: longer tracks get the same laps' worth.
TICK_CAP_PER_OPT_S = 45.0 / 7.02


def episode_tick_cap(cfg: Config) -> int:
    """Episode cap for this map in plant ticks (1 ms)."""
    return int(round(TICK_CAP_PER_OPT_S * float(cfg.track.opt_time) * 1000.0))


def control_dt() -> float:
    """Control period in seconds from the car parameters."""
    return float(car_kwargs()["control_period"]) / 1000.0


def env_kwargs(cfg: Config) -> dict[str, float]:
    """Car parameters plus this run's rewards, pace target, and speed cap."""
    kwargs = car_kwargs()
    if cfg.track.max_speed_mps is not None:
        kwargs["v_max"] = float(cfg.track.max_speed_mps)
    kwargs.update(
        {
            "progress_reward": float(cfg.ppo.r_progress),
            "collision_reward": float(cfg.ppo.r_collision),
            "lap_completion_reward": float(cfg.ppo.r_lap),
            "optimal_lap_time": float(cfg.track.opt_time),
        }
    )
    return kwargs


def make_venv(cfg: Config, n: int, seed: int, sim2real: bool | None = None) -> VecEnv:
    """Construct a batched plant for this config."""
    if sim2real is None:
        sim2real = bool(cfg.training.sim2real)
    return VecEnv(
        num_envs=n,
        blob_path=str(ensure_blob(cfg.track.map)),
        seed=seed,
        sim2real=sim2real,
        **env_kwargs(cfg),
    )


def mean_actions(agent: PPOAgent, states: np.ndarray) -> np.ndarray:
    """Actor-mean actions for a batch of observations."""
    st = T.tensor(states, dtype=T.float32).to(agent.actor.device)
    with T.no_grad():
        return agent.actor(st).mean.cpu().numpy().astype(np.float32, copy=False)


@dataclass
class LayoutRun:
    """Mean-action outcome of one layout in a batched plant run."""

    success: bool
    collided: bool
    lap_times: list[float]
    steps: int
    terminal_v: float


def run_layouts(
    agent: PPOAgent,
    cfg: Config,
    layouts: Sequence[Layout],
    seed: int,
    sim2real: bool | None = None,
    num_laps: int | None = None,
) -> list[LayoutRun]:
    """Run mean-action flying laps, one env per layout."""
    n_layouts = len(layouts)
    if n_layouts == 0:
        return []
    num_laps = int(cfg.training.num_laps if num_laps is None else num_laps)
    max_steps = int(EVAL_TIMEOUT_FACTOR * cfg.track.opt_time * max(1, num_laps) / control_dt())
    venv = make_venv(cfg, n_layouts, seed, sim2real=sim2real)
    try:
        venv.reset()
        tick_cap = episode_tick_cap(cfg)
        for i, layout in enumerate(layouts):
            venv.set_lap_cap(i, num_laps + 1)
            venv.set_tick_cap(i, tick_cap)
            venv.reset_env(i)
            venv.set_obstacles(i, to_sim(layout))
        states = venv.obs.copy()

        finished = [False] * n_layouts
        success = [False] * n_layouts
        collided = [False] * n_layouts
        per_lap_times: list[list[float]] = [[] for _ in range(n_layouts)]
        prev_lap_count = [0] * n_layouts
        steps = np.zeros(n_layouts, dtype=np.int32)
        terminal_v = np.zeros(n_layouts, dtype=np.float64)

        for _ in range(max_steps):
            actions = np.zeros((n_layouts, 2), dtype=np.float32)
            live = [k for k in range(n_layouts) if not finished[k]]
            if live:
                actions[live] = mean_actions(agent, states[live])
                steps[live] += 1
            states, rewards, terminals = venv.step(actions)

            for k in range(n_layouts):
                if finished[k]:
                    continue
                if terminals[k]:
                    kind = venv.done_kind(k)
                    success[k] = (
                        kind == COMPLETE
                        and venv.done_laps(k) >= num_laps + 1
                        and float(rewards[k]) > 0.0
                    )
                    if success[k]:
                        per_lap_times[k].append(float(venv.done_lap_time(k)))
                    collided[k] = kind == COLLISION
                    finished[k] = True
                    continue
                terminal_v[k] = float(venv.vehicle_state(k)[4])
                cur_lap = venv.current_lap(k)
                if cur_lap > prev_lap_count[k]:
                    prev_lap_count[k] = cur_lap
                    # The first completed lap is the standing start; later ones are flying laps.
                    if cur_lap >= 2:
                        per_lap_times[k].append(float(venv.last_lap_time(k)))
            if all(finished):
                break
    finally:
        venv.close()

    return [
        LayoutRun(
            success=bool(success[k]),
            collided=bool(collided[k]),
            lap_times=per_lap_times[k],
            steps=int(steps[k]),
            terminal_v=float(terminal_v[k]),
        )
        for k in range(n_layouts)
    ]


def evaluate(agent: PPOAgent, cfg: Config, layouts: Sequence[Layout], seed: int) -> dict[str, float]:
    """Run mean-action flying laps, one env per layout, and return success rate and pace."""
    runs = run_layouts(agent, cfg, layouts, seed)
    if not runs:
        return {"success_rate": 0.0, "pace": 0.0}
    completed_means = [float(np.mean(run.lap_times)) for run in runs if run.success and run.lap_times]
    pace = float(cfg.track.opt_time) / float(np.mean(completed_means)) if completed_means else 0.0
    return {
        "success_rate": float(sum(1 for run in runs if run.success) / len(runs)),
        "pace": pace,
    }
