# torch stubs don't mark re-exports (T.tensor, T.clamp, ...)
# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch as T
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_


@dataclass
class RolloutBuffer:
    """Stores one PPO rollout before it is copied into the replay memory."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    probs: list[T.Tensor] = field(default_factory=list)
    vals: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def to_np_array(self) -> dict[str, Any]:
        """Stack buffer fields into numpy arrays."""
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "probs": np.array([t.cpu().numpy() for t in self.probs]),
            "vals": np.array(self.vals),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
        }

    def clear_data(self) -> None:
        """Drop every stored transition."""
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()


class PPOMemory:
    """Holds the current rollout and yields shuffled minibatch indices."""

    def __init__(self, batch_size: int) -> None:
        self.rollout_buffer = RolloutBuffer()
        self.batch_size = batch_size

    def generate_batches(self) -> tuple[dict[str, Any], list[np.ndarray]]:
        """Return the stacked rollout and shuffled minibatch index slices."""
        n_states = len(self.rollout_buffer.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        return self.rollout_buffer.to_np_array(), batches

    def store_memory(self, rollout_buffer: RolloutBuffer) -> None:
        """Append another buffer's transitions onto this memory."""
        self.rollout_buffer.states.extend(rollout_buffer.states)
        self.rollout_buffer.actions.extend(rollout_buffer.actions)
        self.rollout_buffer.probs.extend(rollout_buffer.probs)
        self.rollout_buffer.vals.extend(rollout_buffer.vals)
        self.rollout_buffer.rewards.extend(rollout_buffer.rewards)
        self.rollout_buffer.dones.extend(rollout_buffer.dones)

    def clear_memory(self) -> None:
        """Clear the stored rollout."""
        self.rollout_buffer.clear_data()


class ActorNetwork(nn.Module):
    """Two-layer tanh MLP that parameterises a diagonal Gaussian policy."""

    def __init__(
        self,
        n_actions: int,
        input_dims: int,
        alpha: float,
        log_std_init: float,
        actor_file: Path,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
    ) -> None:
        super().__init__()
        self.path = actor_file
        flat_input_dims = int(np.prod(input_dims))
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, n_actions),
            nn.Tanh(),
        )
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)  # type: ignore[arg-type]
        self.log_std = nn.Parameter(T.full((1, n_actions), log_std_init))
        self.log_std_max = 2.0
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> Normal:
        """Return the action distribution for a batch of states."""
        mean = self.actor(state)
        clamped_log_std = T.clamp(self.log_std, min=-20, max=self.log_std_max)
        std = T.clamp(T.exp(clamped_log_std), min=1e-6)
        return Normal(mean, std)

    def save_checkpoint(self) -> None:
        """Write the actor weights to disk."""
        T.save(self.state_dict(), self.path)

    def load_checkpoint(self) -> None:
        """Read the actor weights from disk."""
        self.load_state_dict(T.load(self.path, map_location=self.device))


class CriticNetwork(nn.Module):
    """Two-layer tanh MLP that estimates state value."""

    def __init__(
        self,
        input_dims: int,
        alpha: float,
        critic_file: Path,
        fc1_dims: int = 256,
        fc2_dims: int = 256,
    ) -> None:
        super().__init__()
        self.path = critic_file
        flat_input_dims = int(np.prod(input_dims))
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_input_dims, fc1_dims),
            nn.Tanh(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Tanh(),
            nn.Linear(fc2_dims, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cpu")
        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Return V(s) for a batch of states."""
        return self.critic(state)

    def save_checkpoint(self) -> None:
        """Write the critic weights to disk."""
        T.save(self.state_dict(), self.path)

    def load_checkpoint(self) -> None:
        """Read the critic weights from disk."""
        self.load_state_dict(T.load(self.path, map_location=self.device))


class PPOAgent:
    """Actor-critic PPO with GAE advantages and CAPS action smoothing."""

    def __init__(
        self,
        n_actions: int,
        input_dims: int,
        gamma: float,
        alpha: float,
        gae_lambda: float,
        policy_clip: float,
        batch_size: int,
        n_epochs: int,
        log_std_init: float,
        initial_c2: float,
        normalise_advantages: bool,
        target_kl: float,
        c1: float,
        base_path: Path | str,
        run_name: str,
        seed: int,
        hidden_size: int = 256,
        caps_lambda_t: float = 1.0,
        caps_lambda_s: float = 0.5,
        caps_sigma_s: float = 0.05,
        caps_lambda_t_v: float = 0.25,
        caps_lambda_s_v: float = 0.5,
    ) -> None:
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.target_kl = target_kl
        self.c1 = c1
        self.c2 = initial_c2
        self.normalise_advantages = normalise_advantages

        self.caps_lambda_t = caps_lambda_t
        self.caps_lambda_s = caps_lambda_s
        self.caps_lambda_t_v = caps_lambda_t_v
        self.caps_lambda_s_v = caps_lambda_s_v
        self.caps_sigma_s = caps_sigma_s
        self._caps_lambda_t_base = caps_lambda_t
        self._caps_lambda_s_base = caps_lambda_s
        self._caps_lambda_t_v_base = caps_lambda_t_v
        self._caps_lambda_s_v_base = caps_lambda_s_v

        critic_file = os.path.join(base_path, run_name, f"_critic_seed_{seed}")
        actor_file = os.path.join(base_path, run_name, f"_actor_seed_{seed}")
        os.makedirs(os.path.dirname(actor_file), exist_ok=True)

        self.actor = ActorNetwork(
            n_actions,
            input_dims,
            alpha,
            log_std_init,
            Path(actor_file),
            fc1_dims=hidden_size,
            fc2_dims=hidden_size,
        )
        self.critic = CriticNetwork(
            input_dims,
            alpha,
            Path(critic_file),
            fc1_dims=hidden_size,
            fc2_dims=hidden_size,
        )
        self.memory = PPOMemory(batch_size)
        self.rollout_buffer = RolloutBuffer()

    def set_c2(self, new_c2_value: float) -> None:
        """Set the entropy bonus weight."""
        self.c2 = new_c2_value

    def set_caps_scale(self, scale: float) -> None:
        """Scale CAPS weights by a factor in [0, 1]."""
        self.caps_lambda_t = self._caps_lambda_t_base * scale
        self.caps_lambda_s = self._caps_lambda_s_base * scale
        self.caps_lambda_t_v = self._caps_lambda_t_v_base * scale
        self.caps_lambda_s_v = self._caps_lambda_s_v_base * scale

    def store_rollout_buffer(self) -> None:
        """Move the live buffer into memory and clear it."""
        self.memory.store_memory(self.rollout_buffer)
        self.rollout_buffer.clear_data()

    def save_models(self) -> None:
        """Save actor and critic to their current paths."""
        print("...saving models")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def save_models_with_suffix(self, suffix: str) -> None:
        """Save actor and critic next to the current paths with a name suffix."""
        print(f"...saving models with suffix '{suffix}'")
        original_actor_path = self.actor.path
        original_critic_path = self.critic.path
        actor_path_obj = Path(original_actor_path)
        critic_path_obj = Path(original_critic_path)
        self.actor.path = (
            actor_path_obj.parent / f"{actor_path_obj.stem}{suffix}{actor_path_obj.suffix}"
        )
        self.critic.path = (
            critic_path_obj.parent / f"{critic_path_obj.stem}{suffix}{critic_path_obj.suffix}"
        )
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.actor.path = original_actor_path
        self.critic.path = original_critic_path

    def choose_action_batched(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, T.Tensor, T.Tensor]:
        """Sample actions for a batch of observations without touching the buffer."""
        states = T.tensor(observations, dtype=T.float32).to(self.actor.device)
        dist = self.actor(states)
        values = self.critic(states)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=1)
        return (
            actions.cpu().detach().numpy(),
            log_probs.detach(),
            values.squeeze(1).detach(),
        )

    def store_transition(
        self,
        state: np.ndarray,
        action_np: np.ndarray,
        log_prob_tensor: T.Tensor,
        val: float,
        reward: float,
        done: bool,
    ) -> None:
        """Append one env-step to the live rollout buffer."""
        self.rollout_buffer.states.append(state)
        self.rollout_buffer.actions.append(action_np)
        self.rollout_buffer.probs.append(log_prob_tensor)
        self.rollout_buffer.vals.append(val)
        self.rollout_buffer.rewards.append(reward)
        self.rollout_buffer.dones.append(done)

    def learn(
        self, next_observations: np.ndarray, next_dones: list[bool] | np.ndarray, n_envs: int = 1
    ) -> dict[str, float]:
        """Run PPO epochs over the stored interleaved rollout and clear memory."""
        entropies: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        kls: list[float] = []

        # generate_batches also shuffles with np.random; seeded runs depend on that draw.
        traj, _ = self.memory.generate_batches()
        states_arr = traj["states"]
        actions_arr = traj["actions"]
        old_probs_arr = traj["probs"]
        rewards_arr = traj["rewards"]
        dones_arr = traj["dones"]
        n_transitions = len(rewards_arr)

        idx = np.arange(n_transitions)
        next_idx = idx + n_envs
        caps_valid = (next_idx < n_transitions) & (~dones_arr.astype(bool))
        next_idx_c = np.where(caps_valid, next_idx, idx)
        next_states_arr = states_arr[next_idx_c]
        caps_mask_arr = caps_valid.astype(np.float32)

        assert n_transitions % n_envs == 0, "rollout not a multiple of n_envs"
        t_steps = n_transitions // n_envs
        rewards_2d = rewards_arr.reshape(t_steps, n_envs)
        dones_2d = dones_arr.reshape(t_steps, n_envs)

        all_states_tensor = T.tensor(states_arr, dtype=T.float32).to(self.critic.device)
        all_actions_tensor = T.tensor(actions_arr, dtype=T.float32).to(self.actor.device)
        all_old_probs_tensor = T.tensor(old_probs_arr, dtype=T.float32).to(self.actor.device)
        all_caps_next_tensor = T.tensor(next_states_arr, dtype=T.float32).to(self.actor.device)
        all_caps_mask_tensor = T.tensor(caps_mask_arr, dtype=T.float32).to(self.actor.device)
        next_obs_tensor = T.tensor(np.asarray(next_observations), dtype=T.float32).to(
            self.critic.device
        )
        next_not_done = 1.0 - np.asarray(next_dones, dtype=np.float32)

        kl_exceeded = False
        for _ in range(self.n_epochs):
            with T.no_grad():
                fresh_vals = self.critic(all_states_tensor).squeeze(1).cpu().numpy()
                next_vals = self.critic(next_obs_tensor).squeeze(1).cpu().numpy() * next_not_done

            all_advantages = self.compute_gae(
                rewards_2d, dones_2d, fresh_vals.reshape(t_steps, n_envs), next_vals
            ).reshape(n_transitions)

            # Advantages are recomputed each epoch from the current critic; the ratio still clips against old_probs.
            raw_advantages = T.tensor(all_advantages, dtype=T.float32).to(self.actor.device)
            advantages = raw_advantages
            if self.normalise_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            values = T.tensor(fresh_vals, dtype=T.float32).to(self.actor.device)
            # The lambda-return is raw advantage + V(s); normalisation is for
            # the policy gradient only.
            returns = (raw_advantages + values).detach()

            indices = np.arange(n_transitions, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [
                indices[j : j + self.memory.batch_size]
                for j in range(0, n_transitions, self.memory.batch_size)
            ]

            for batch in batches:
                batch_idx = T.from_numpy(batch).to(self.actor.device)
                states = all_states_tensor[batch_idx]
                actions = all_actions_tensor[batch_idx]
                old_probs = all_old_probs_tensor[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                caps_next = all_caps_next_tensor[batch_idx]
                caps_m = all_caps_mask_tensor[batch_idx]

                entropy, actor_loss, critic_loss, kl, kl_stop = self.sgd_step(
                    states,
                    old_probs,
                    actions,
                    batch_advantages,
                    batch_returns,
                    critic_only=kl_exceeded,
                    caps_next_states=caps_next,
                    caps_mask=caps_m,
                )

                critic_losses.append(critic_loss)
                if entropy is not None:
                    assert actor_loss is not None and kl is not None
                    entropies.append(entropy)
                    actor_losses.append(actor_loss)
                    kls.append(kl)
                if kl_stop and not kl_exceeded:
                    kl_exceeded = True
                    print(
                        f"KL {kl:.4f} exceeded target {self.target_kl} — actor frozen, critic continues"
                    )

        self.memory.clear_memory()
        return {
            "entropy": float(np.mean(entropies) if entropies else 0.0),
            "actor_loss": float(np.mean(actor_losses) if actor_losses else 0.0),
            "critic_loss": float(np.mean(critic_losses) if critic_losses else 0.0),
            "approx_kl": float(np.mean(kls) if kls else 0.0),
        }

    def compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: np.ndarray | float,
    ) -> np.ndarray:
        """Compute GAE over time, vectorised across workers when rewards are 2-D."""
        rewards = np.asarray(rewards, dtype=np.float32)
        advantage = np.zeros_like(rewards)
        gae = np.zeros(rewards.shape[1:], dtype=np.float32)
        next_row = np.asarray(next_value, dtype=np.float32).reshape((1,) + rewards.shape[1:])
        vals_with_next = np.concatenate(
            [values.astype(np.float32, copy=False), next_row], axis=0
        )
        for t in reversed(range(len(rewards))):
            terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * vals_with_next[t + 1] * terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * terminal * gae
            advantage[t] = gae
        return advantage

    def sgd_step(
        self,
        states: T.Tensor,
        old_probs: T.Tensor,
        actions: T.Tensor,
        advantages: T.Tensor,
        returns: T.Tensor,
        critic_only: bool = False,
        caps_next_states: T.Tensor | None = None,
        caps_mask: T.Tensor | None = None,
    ) -> tuple[float | None, float | None, float, float | None, bool]:
        """Take one actor-critic SGD step, or critic-only after a KL stop."""
        critic_values = self.critic(states).squeeze(-1)
        critic_loss = F.mse_loss(critic_values, returns)

        kl: float | None = None
        kl_stop = False
        dist: Normal | None = None
        new_log_probs: T.Tensor | None = None
        if not critic_only:
            action_dist: Normal = self.actor(states)
            dist = action_dist
            new_log_probs = action_dist.log_prob(actions).sum(dim=1)
            with T.no_grad():
                log_ratio = new_log_probs - old_probs
                kl = ((T.exp(log_ratio) - 1.0) - log_ratio).mean().item()
            if kl > self.target_kl:
                critic_only = True
                kl_stop = True

        if critic_only:
            self.critic.optimizer.zero_grad()
            (self.c1 * critic_loss).backward()
            clip_grad_norm_(self.critic.parameters(), 0.2)
            self.critic.optimizer.step()
            return None, None, critic_loss.item(), kl, kl_stop

        assert dist is not None and new_log_probs is not None
        prob_ratio = T.exp(new_log_probs - old_probs)
        weighted_probs = advantages * prob_ratio
        clipped_ratio = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
        weighted_clipped_probs = clipped_ratio * advantages
        actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
        entropy = dist.entropy().mean()

        assert caps_next_states is not None and caps_mask is not None
        mean = dist.mean
        eps = T.randn_like(states) * self.caps_sigma_s
        mean_p, mean_n = self.actor.actor(T.cat([states + eps, caps_next_states])).chunk(2)
        sp = ((mean - mean_p) ** 2).mean(dim=0)
        caps_s_t = self.caps_lambda_s * sp[0] + self.caps_lambda_s_v * sp[1]
        denom = caps_mask.sum().clamp(min=1.0)
        tp = (caps_mask.unsqueeze(1) * (mean - mean_n) ** 2).sum(dim=0) / denom
        caps_t_t = self.caps_lambda_t * tp[0] + self.caps_lambda_t_v * tp[1]
        caps_loss = caps_t_t + caps_s_t

        total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy + caps_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.actor.parameters(), 0.2)
        clip_grad_norm_(self.critic.parameters(), 0.2)
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return entropy.item(), actor_loss.item(), critic_loss.item(), kl, False
