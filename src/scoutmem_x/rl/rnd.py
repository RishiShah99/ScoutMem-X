"""Random Network Distillation (RND) intrinsic rewards for curiosity-driven exploration.

Implements Burda et al. 2018: a fixed random target network and a trainable
predictor network. The prediction error (MSE) serves as an intrinsic reward
bonus — high error signals novel states, driving exploration into unvisited
regions of the grid.

Usage:
    python -m scoutmem_x.rl.rnd --timesteps 200000
    python -m scoutmem_x.rl.rnd --timesteps 500000 --rnd-coef 0.2 --seed 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv


class _RNDNetwork(nn.Module):
    """Simple MLP shared by both target and predictor networks."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, output_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDRewardWrapper(gym.Wrapper):
    """Gymnasium wrapper that adds RND intrinsic reward to each step.

    The intrinsic bonus = rnd_coef * MSE(predictor(obs), target(obs)).
    The predictor is trained online with one gradient step per environment step.
    The target network is frozen (never updated).

    Args:
        env: Base gymnasium environment.
        rnd_coef: Scaling factor for the intrinsic reward (default 0.1).
        lr: Learning rate for the predictor optimizer.
        input_dim: Observation dimension (frame_stack * frame_dim = 4 * 16 = 64).
    """

    def __init__(
        self,
        env: gym.Env,
        rnd_coef: float = 0.1,
        lr: float = 1e-3,
        input_dim: int = 64,
    ) -> None:
        super().__init__(env)
        self.rnd_coef = rnd_coef

        self._device = torch.device("cpu")

        # Fixed random target — NEVER trained
        self._target = _RNDNetwork(input_dim=input_dim).to(self._device)
        for p in self._target.parameters():
            p.requires_grad = False
        self._target.eval()

        # Trainable predictor
        self._predictor = _RNDNetwork(input_dim=input_dim).to(self._device)
        self._optimizer = torch.optim.Adam(self._predictor.parameters(), lr=lr)
        self._mse = nn.MSELoss()

    def _compute_intrinsic_reward(self, obs: np.ndarray) -> float:
        """Compute RND bonus and train predictor with one gradient step."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)

        with torch.no_grad():
            target_out = self._target(obs_t)

        pred_out = self._predictor(obs_t)
        loss = self._mse(pred_out, target_out)

        # One gradient step on the predictor
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self._compute_intrinsic_reward(obs)
        combined_reward = extrinsic_reward + self.rnd_coef * intrinsic_reward

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["rnd_bonus"] = self.rnd_coef * intrinsic_reward

        return obs, combined_reward, terminated, truncated, info


def _make_rnd_env(rnd_coef: float = 0.1, seed: int = 0) -> RNDRewardWrapper:
    """Factory for RND-wrapped hard environment."""
    base = ScoutMemEnv.hard()
    base.reset(seed=seed)
    return RNDRewardWrapper(base, rnd_coef=rnd_coef)


def _make_eval_env(seed: int = 0) -> ScoutMemEnv:
    """Factory for clean eval environment (no RND wrapper)."""
    env = ScoutMemEnv.hard()
    env.reset(seed=seed)
    return env


def train_rnd(
    timesteps: int = 200_000,
    output_dir: str = "outputs/rl_rnd",
    seed: int = 42,
    rnd_coef: float = 0.1,
) -> Path:
    """Train PPO with RND intrinsic rewards on the hard environment.

    Evaluation is done on the hard environment WITHOUT the RND wrapper,
    so success rate reflects true extrinsic performance.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training PPO + RND for {timesteps:,} timesteps (rnd_coef={rnd_coef}, seed={seed})")
    print(f"Output: {out}")

    np.random.seed(seed)

    # Training env: hard + RND wrapper
    train_env = make_vec_env(
        lambda: _make_rnd_env(rnd_coef=rnd_coef, seed=seed), n_envs=8,
    )
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_reward=5.0,
    )

    # Eval env: hard without RND (extrinsic reward only)
    eval_env = make_vec_env(lambda: _make_eval_env(seed=seed + 1000), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        tensorboard_log=None,
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "best_model"),
        log_path=str(out / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)

    model.save(str(out / "final_model"))
    train_env.save(str(out / "vec_normalize.pkl"))
    print(f"Model saved to {out / 'final_model.zip'}")

    return out / "final_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ScoutMem-X RL policy with RND rewards")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--output", type=str, default="outputs/rl_rnd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rnd-coef", type=float, default=0.1,
        help="Intrinsic reward scaling factor (default: 0.1)",
    )
    args = parser.parse_args()

    model_path = train_rnd(
        timesteps=args.timesteps,
        output_dir=args.output,
        seed=args.seed,
        rnd_coef=args.rnd_coef,
    )

    # Quick eval on hard without RND wrapper
    from scoutmem_x.rl.train import evaluate

    print("\nEvaluating trained model (extrinsic reward only, hard)...")
    metrics = evaluate(str(model_path.with_suffix("")), difficulty="hard")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
