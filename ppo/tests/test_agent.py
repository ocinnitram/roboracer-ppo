# torch stubs don't mark re-exports (T.manual_seed)
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import os
import tempfile

import numpy as np
import torch as T

from ppo.agent import PPOAgent

os.environ.setdefault("OMP_NUM_THREADS", "1")


def make_agent(tmp: str, dim: int) -> PPOAgent:
    """Build a small agent for the tests."""
    return PPOAgent(
        n_actions=2,
        input_dims=dim,
        gamma=0.995,
        alpha=3e-4,
        gae_lambda=0.97,
        policy_clip=0.2,
        batch_size=32,
        n_epochs=2,
        log_std_init=-0.7,
        initial_c2=0.01,
        normalise_advantages=False,
        target_kl=0.03,
        c1=0.5,
        base_path=tmp,
        run_name="agent_test",
        seed=0,
        hidden_size=64,
    )


def gae_reference(
    agent: PPOAgent,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
) -> np.ndarray:
    """Scalar-loop GAE used as the oracle for the vectorised implementation."""
    advantage = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    vals_with_next = np.append(values, next_value)
    for t in reversed(range(len(rewards))):
        terminal = 1 - dones[t]
        delta = rewards[t] + agent.gamma * vals_with_next[t + 1] * terminal - values[t]
        gae = delta + agent.gamma * agent.gae_lambda * terminal * gae
        advantage[t] = gae
    return advantage


def test_gae_equivalence() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_agent(tmp, dim=26)
        t_steps, n_envs = 64, 8
        rewards = rng.standard_normal((t_steps, n_envs)).astype(np.float32)
        dones = rng.random((t_steps, n_envs)) < 0.05
        values = rng.standard_normal((t_steps, n_envs)).astype(np.float32)
        next_vals = rng.standard_normal(n_envs).astype(np.float32)

        vec = agent.compute_gae(rewards, dones, values, next_vals)
        for i in range(n_envs):
            ref = gae_reference(
                agent, rewards[:, i], dones[:, i], values[:, i], next_vals[i]
            )
            assert np.allclose(vec[:, i], ref, atol=1e-6), f"GAE mismatch worker {i}"
        ref0 = gae_reference(
            agent, rewards[:, 0], dones[:, 0], values[:, 0], float(next_vals[0])
        )
        one_d = agent.compute_gae(
            rewards[:, 0], dones[:, 0], values[:, 0], float(next_vals[0])
        )
        assert np.allclose(one_d, ref0, atol=1e-6)


def test_learn_smoke() -> None:
    rng = np.random.default_rng(0)
    T.manual_seed(0)
    dim = 26
    with tempfile.TemporaryDirectory() as tmp:
        agent = make_agent(tmp, dim=dim)
        n_envs, t_steps = 4, 32
        for _ in range(t_steps):
            obs = rng.standard_normal((n_envs, dim)).astype(np.float32)
            actions_np, log_probs, vals = agent.choose_action_batched(obs)
            for i in range(n_envs):
                agent.store_transition(
                    obs[i],
                    actions_np[i],
                    log_probs[i],
                    vals[i].item(),
                    float(rng.standard_normal()),
                    bool(rng.random() < 0.05),
                )
        agent.store_rollout_buffer()
        diag = agent.learn(
            rng.standard_normal((n_envs, dim)).astype(np.float32),
            [False] * n_envs,
            n_envs,
        )
        for k, v in diag.items():
            assert np.isfinite(v), f"non-finite diagnostic {k}={v}"
        assert len(agent.memory.rollout_buffer.rewards) == 0, "memory not cleared"
