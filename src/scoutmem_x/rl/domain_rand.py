"""Domain randomization wrapper for ScoutMem-X RL training.

Randomizes environment parameters per episode to train a more robust policy
that generalizes across grid sizes, noise levels, and object configurations.

Usage:
    python -m scoutmem_x.rl.domain_rand --timesteps 200000
    python -m scoutmem_x.rl.domain_rand --eval --eval-model outputs/rl_domrand/best_model/best_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv


class DomainRandomizationWrapper(gym.Wrapper):
    """Randomizes ScoutMemEnv parameters on each reset for sim-to-sim transfer.

    Since ScoutMemEnv normalizes positions by grid_size internally, the 16-dim
    per-frame observation is always [0,1] bounded regardless of grid_size.
    This means observation_space and action_space stay consistent across resets.
    """

    def __init__(
        self,
        env: ScoutMemEnv,
        *,
        grid_sizes: list[int] | None = None,
        n_objects_range: list[int] | None = None,
        n_distractors_range: list[int] | None = None,
        dropout_range: tuple[float, float] = (0.03, 0.15),
        noise_range: tuple[float, float] = (0.02, 0.10),
        view_range_range: tuple[float, float] = (1.5, 2.5),
    ) -> None:
        super().__init__(env)
        self._frame_stack = env.frame_stack
        self._grid_sizes = grid_sizes or [3, 4, 5]
        self._n_objects_range = n_objects_range or [3, 4, 5, 6]
        self._n_distractors_range = n_distractors_range or [0, 1, 2]
        self._dropout_range = dropout_range
        self._noise_range = noise_range
        self._view_range_range = view_range_range

        # Lock spaces from the initial env -- they never change
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        rng = np.random.default_rng(seed)

        grid_size = int(rng.choice(self._grid_sizes))
        n_objects = int(rng.choice(self._n_objects_range))
        n_distractors = int(rng.choice(self._n_distractors_range))
        dropout_rate = float(rng.uniform(*self._dropout_range))
        noise_std = float(rng.uniform(*self._noise_range))
        view_range = float(rng.uniform(*self._view_range_range))

        # Clamp n_objects + n_distractors to not exceed grid cells
        max_entities = grid_size * grid_size
        if n_objects + n_distractors > max_entities:
            n_objects = min(n_objects, max_entities)
            n_distractors = min(n_distractors, max_entities - n_objects)

        # Build a fresh inner env with randomized params
        new_env = ScoutMemEnv(
            grid_size=grid_size,
            n_objects=n_objects,
            n_distractors=n_distractors,
            view_range=view_range,
            dropout_rate=dropout_rate,
            noise_std=noise_std,
            frame_stack=self._frame_stack,
        )
        self.env = new_env

        return self.env.reset(seed=seed, options=options)


def train_domain_rand(
    timesteps: int = 200_000,
    output_dir: str = "outputs/rl_domrand",
    seed: int = 42,
) -> Path:
    """Train PPO with domain randomization for generalization."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Domain-randomized training: {timesteps:,} steps, seed={seed}")
    print(f"  grid_sizes=[3,4,5]  dropout=[0.03,0.15]  noise=[0.02,0.10]")
    print(f"Output: {out}")

    np.random.seed(seed)

    def make_train_env() -> DomainRandomizationWrapper:
        base = ScoutMemEnv.hard()
        return DomainRandomizationWrapper(base)

    def make_eval_env() -> ScoutMemEnv:
        env = ScoutMemEnv.hard()
        env.reset(seed=seed + 1000)
        return env

    train_env = make_vec_env(make_train_env, n_envs=8)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_reward=5.0)

    eval_env = make_vec_env(make_eval_env, n_envs=1)
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


def evaluate_domain_rand(
    model_path: str,
    n_episodes: int = 100,
    vec_normalize_path: str | None = None,
) -> dict[str, float]:
    """Evaluate a domain-randomized model on fixed hard difficulty."""
    model = PPO.load(model_path)

    eval_venv = make_vec_env(lambda: ScoutMemEnv.hard(), n_envs=1)

    if vec_normalize_path is None:
        candidate = Path(model_path).parent.parent / "vec_normalize.pkl"
        if candidate.exists():
            vec_normalize_path = str(candidate)

    if vec_normalize_path and Path(vec_normalize_path).exists():
        eval_venv = VecNormalize.load(vec_normalize_path, eval_venv)
        eval_venv.training = False
        eval_venv.norm_reward = False
        print(f"  Loaded VecNormalize from {vec_normalize_path}")

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for _ in range(n_episodes):
        obs = eval_venv.reset()
        ep_reward = 0.0

        for step in range(25):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = eval_venv.step(action)
            ep_reward += float(rewards[0])

            if dones[0]:
                if float(rewards[0]) > 0:
                    successes += 1
                total_steps += step + 1
                break
        else:
            total_steps += 25

        total_reward += ep_reward

    eval_venv.close()
    return {
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain-randomized RL training for ScoutMem-X")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--output", type=str, default="outputs/rl_domrand")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--eval-model", type=str, default=None,
                        help="Path to model for evaluation")
    args = parser.parse_args()

    if args.eval:
        model_path = args.eval_model or f"{args.output}/final_model"
        print(f"Evaluating domain-rand model: {model_path}")
        metrics = evaluate_domain_rand(model_path)
        print(json.dumps(metrics, indent=2))
    else:
        train_domain_rand(
            timesteps=args.timesteps,
            output_dir=args.output,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
