"""Batched mean-action evaluation of one agent on a list of layouts."""

from __future__ import annotations

from collections.abc import Sequence

from evaluation.common import COMPLETED, CRASHED, TIMEOUT, LayoutResult
from ppo.agent import PPOAgent
from ppo.config import Config
from ppo.layouts import Layout
from ppo.rollout import EVAL_SEED, control_dt, run_layouts


def run(
    agent: PPOAgent,
    cfg: Config,
    layouts: Sequence[Layout],
    *,
    sim2real: bool,
    num_laps: int,
) -> list[LayoutResult]:
    """Run every layout in one plant seeded with EVAL_SEED."""
    dt = control_dt()
    records = run_layouts(agent, cfg, layouts, EVAL_SEED, sim2real=sim2real, num_laps=num_laps)
    results: list[LayoutResult] = []
    for index, (layout, rec) in enumerate(zip(layouts, records, strict=True)):
        if rec.success:
            outcome = COMPLETED
        elif rec.collided:
            outcome = CRASHED
        else:
            outcome = TIMEOUT
        results.append(
            LayoutResult(
                index=index,
                n_obstacles=len(layout),
                outcome=outcome,
                lap_times=list(rec.lap_times),
                steps=rec.steps,
                sim_time=rec.steps * dt,
                terminal_v=rec.terminal_v,
            )
        )
    return results
