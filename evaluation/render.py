"""Watch one policy drive the eval layouts in real time in the raylib window."""

from __future__ import annotations

import time
from collections.abc import Sequence

from ppo.agent import PPOAgent
from ppo.config import Config
from ppo.layouts import Layout, to_sim
from ppo.rollout import (
    EVAL_SEED,
    EVAL_TIMEOUT_FACTOR,
    control_dt,
    episode_tick_cap,
    make_venv,
    mean_actions,
)
from sim.env import COLLISION, COMPLETE
from sim.render import COMPLETED, CRASHED, TIMEOUT, Window

# Seconds the final frame of each layout stays on screen.
HOLD_S = 1.0


def run(
    agent: PPOAgent,
    cfg: Config,
    layouts: Sequence[Layout],
    *,
    first: int,
    sim2real: bool,
    num_laps: int,
) -> None:
    """Drive layouts[first:] one at a time until the window is closed."""
    dt = control_dt()
    max_steps = int(EVAL_TIMEOUT_FACTOR * cfg.track.opt_time * max(1, num_laps) / dt)
    window = Window()
    venv = make_venv(cfg, 1, EVAL_SEED, sim2real=sim2real)
    try:
        for index in range(first, len(layouts)):
            label = f"layout {index}" if layouts[index] else "empty track"
            # Same plant seed and reset sequence as env `index` of the scored eval.
            venv.seed(0, EVAL_SEED * 1000 + index)
            venv.reset()
            venv.set_lap_cap(0, num_laps + 1)
            venv.set_tick_cap(0, episode_tick_cap(cfg))
            venv.reset_env(0)
            venv.set_obstacles(0, to_sim(layouts[index]))
            states = venv.obs.copy()
            outcome = TIMEOUT
            if not window.draw(venv, 0, label):
                return
            start = time.perf_counter()
            for step in range(max_steps):
                states, rewards, terminals = venv.step(mean_actions(agent, states))
                if terminals[0]:
                    kind = venv.done_kind(0)
                    if kind == COMPLETE and venv.done_laps(0) >= num_laps + 1 and float(rewards[0]) > 0.0:
                        outcome = COMPLETED
                    elif kind == COLLISION:
                        outcome = CRASHED
                    break
                if not window.draw(venv, 0, label):
                    return
                wait = start + (step + 1) * dt - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            print(f"{label}: {('timeout', 'crashed', 'completed')[outcome]}", flush=True)
            end = time.perf_counter() + HOLD_S
            while time.perf_counter() < end:
                if not window.draw(venv, 0, label, outcome):
                    return
    finally:
        venv.close()
        window.close()
