"""Train an RL exploration policy for ScoutMem-X.

Usage:
    python -m scoutmem_x.rl.train --timesteps 500000
    python -m scoutmem_x.rl.train --difficulty easy --timesteps 100000
    python -m scoutmem_x.rl.train --curriculum --timesteps 500000
    python -m scoutmem_x.rl.train --eval --eval-model outputs/rl/best_model/best_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv


def make_env(difficulty: str = "hard", seed: int = 0) -> ScoutMemEnv:
    factories = {"easy": ScoutMemEnv.easy, "medium": ScoutMemEnv.medium, "hard": ScoutMemEnv.hard}
    env = factories[difficulty]()
    env.reset(seed=seed)
    return env


def train(
    timesteps: int = 50_000,
    output_dir: str = "outputs/rl",
    difficulty: str = "easy",
    seed: int = 42,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Training PPO for {timesteps:,} timesteps on {difficulty} difficulty (seed={seed})")
    print(f"Output: {out}")

    np.random.seed(seed)

    train_env = make_vec_env(lambda: make_env(difficulty=difficulty, seed=seed), n_envs=8)
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_reward=5.0,
    )
    eval_env = make_vec_env(lambda: make_env(difficulty=difficulty, seed=seed + 1000), n_envs=1)
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


def train_curriculum(
    timesteps: int = 500_000,
    output_dir: str = "outputs/rl_curriculum",
    seed: int = 42,
) -> Path:
    """Train with curriculum: easy -> medium -> hard sequentially.

    Splits total timesteps across 3 phases (20%/30%/50%), loading the
    previous model each time so knowledge transfers across difficulties.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phases = [
        ("easy", int(timesteps * 0.20)),
        ("medium", int(timesteps * 0.30)),
        ("hard", int(timesteps * 0.50)),
    ]

    print(f"Curriculum training: {timesteps:,} total steps, seed={seed}")
    for diff, steps in phases:
        print(f"  Phase {diff}: {steps:,} steps")

    model = None
    for i, (difficulty, phase_steps) in enumerate(phases):
        phase_dir = out / difficulty
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Phase {i+1}/3: {difficulty} ({phase_steps:,} steps)")
        print(f"{'='*50}")

        train_env = make_vec_env(
            lambda d=difficulty: make_env(difficulty=d, seed=seed), n_envs=8,
        )
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=False, clip_reward=5.0,
        )
        eval_env = make_vec_env(
            lambda d=difficulty: make_env(difficulty=d, seed=seed + 1000), n_envs=1,
        )
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, training=False,
        )

        if model is None:
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
        else:
            model.set_env(train_env)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(phase_dir / "best_model"),
            log_path=str(phase_dir / "eval_logs"),
            eval_freq=5000,
            n_eval_episodes=30,
            deterministic=True,
        )

        model.learn(
            total_timesteps=phase_steps,
            callback=eval_cb,
            progress_bar=False,
            reset_num_timesteps=False,
        )

        model.save(str(phase_dir / "final_model"))
        train_env.save(str(phase_dir / "vec_normalize.pkl"))
        print(f"  Saved to {phase_dir}")

    # Copy final hard model to top-level
    import shutil
    shutil.copy(str(out / "hard" / "final_model.zip"), str(out / "final_model.zip"))
    shutil.copy(str(out / "hard" / "vec_normalize.pkl"), str(out / "vec_normalize.pkl"))
    print(f"\nCurriculum complete. Final model: {out / 'final_model.zip'}")

    return out / "final_model.zip"


def evaluate(
    model_path: str, n_episodes: int = 100, difficulty: str = "hard",
    vec_normalize_path: str | None = None,
) -> dict[str, float]:
    model = PPO.load(model_path)

    # Wrap in VecNormalize if stats file exists (model was trained with norm_obs=True)
    eval_venv = make_vec_env(lambda: make_env(difficulty=difficulty, seed=123), n_envs=1)
    if vec_normalize_path is None:
        # Try to find vec_normalize.pkl next to the model
        model_dir = Path(model_path).parent
        candidate = model_dir.parent / "vec_normalize.pkl"
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

    for ep in range(n_episodes):
        obs = eval_venv.reset()
        ep_reward = 0.0

        for step in range(25):  # max_steps fallback
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

    return {
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ScoutMem-X RL policy")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--output", type=str, default="outputs/rl")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum learning: easy -> medium -> hard")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-model", type=str, default=None)
    args = parser.parse_args()

    if args.eval:
        model_path = args.eval_model or f"{args.output}/final_model"
        print(f"Evaluating {model_path} on {args.difficulty}...")
        metrics = evaluate(model_path, difficulty=args.difficulty)
        print(json.dumps(metrics, indent=2))
    elif args.curriculum:
        train_curriculum(
            timesteps=args.timesteps,
            output_dir=args.output,
            seed=args.seed,
        )
    else:
        train(
            timesteps=args.timesteps,
            output_dir=args.output,
            difficulty=args.difficulty,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
