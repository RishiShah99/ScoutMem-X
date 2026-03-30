"""Train an RL exploration policy for ScoutMem-X.

Usage:
    python -m scoutmem_x.rl.train --timesteps 500000
    python -m scoutmem_x.rl.train --eval --eval-model outputs/rl/best_model/best_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv


def make_env(seed: int = 0) -> ScoutMemEnv:
    env = ScoutMemEnv(grid_size=5, n_objects=6, n_distractors=2, max_steps=25)
    env.reset(seed=seed)
    return env


def train(timesteps: int = 50_000, output_dir: str = "outputs/rl") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training PPO for {timesteps:,} timesteps...")
    print(f"Output: {out}")

    train_env = make_vec_env(lambda: make_env(seed=42), n_envs=8)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_reward=5.0,
    )
    # Eval env must also be wrapped in VecNormalize to match training env
    eval_env = make_vec_env(lambda: make_env(seed=99), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        tensorboard_log=None,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        },
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "best_model"),
        log_path=str(out / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
    )

    model.learn(
        total_timesteps=timesteps, callback=eval_callback, progress_bar=False,
    )

    model.save(str(out / "final_model"))
    train_env.save(str(out / "vec_normalize.pkl"))
    print(f"Model saved to {out / 'final_model.zip'}")

    return out / "final_model.zip"


def evaluate(model_path: str, n_episodes: int = 100) -> dict[str, float]:
    model = PPO.load(model_path)
    env = make_env(seed=123)

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            ep_reward += reward

            if term:
                if reward > 0:
                    successes += 1
                total_steps += step + 1
                break
            if trunc:
                total_steps += step + 1
                break

        total_reward += ep_reward

    return {
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ScoutMem-X RL policy")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--output", type=str, default="outputs/rl")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-model", type=str, default=None)
    args = parser.parse_args()

    if args.eval:
        model_path = args.eval_model or f"{args.output}/final_model"
        print(f"Evaluating {model_path}...")
        metrics = evaluate(model_path)
        print(json.dumps(metrics, indent=2))
    else:
        train(timesteps=args.timesteps, output_dir=args.output)


if __name__ == "__main__":
    main()
