"""Combined training: RND + Curriculum + Domain Randomization.

Combines all three enhancements into a single training pipeline:
1. Domain randomization wrapper for robustness
2. RND intrinsic rewards for curiosity-driven exploration
3. Sequential curriculum (easy -> medium -> hard) for knowledge transfer

Usage:
    python -m scoutmem_x.rl.combined --timesteps 600000
    python -m scoutmem_x.rl.combined --eval --eval-model outputs/rl_combined/hard/best_model/best_model
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

from scoutmem_x.rl.domain_rand import DomainRandomizationWrapper
from scoutmem_x.rl.env import ScoutMemEnv
from scoutmem_x.rl.rnd import RNDRewardWrapper


def _make_combined_env(difficulty: str = "hard", seed: int = 0, rnd_coef: float = 0.1) -> RNDRewardWrapper:
    """Create env with domain rand + RND."""
    factories = {"easy": ScoutMemEnv.easy, "medium": ScoutMemEnv.medium, "hard": ScoutMemEnv.hard}
    base = factories[difficulty]()
    dr = DomainRandomizationWrapper(base)
    rnd = RNDRewardWrapper(dr, rnd_coef=rnd_coef)
    rnd.reset(seed=seed)
    return rnd


def _make_eval_env(difficulty: str = "hard", seed: int = 0) -> ScoutMemEnv:
    """Clean eval env — no RND, no domain rand."""
    factories = {"easy": ScoutMemEnv.easy, "medium": ScoutMemEnv.medium, "hard": ScoutMemEnv.hard}
    env = factories[difficulty]()
    env.reset(seed=seed)
    return env


def train_combined(
    timesteps: int = 600_000,
    output_dir: str = "outputs/rl_combined",
    seed: int = 42,
    rnd_coef: float = 0.1,
) -> Path:
    """Train with all three enhancements via sequential curriculum."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    phases = [
        ("easy", int(timesteps * 0.15)),
        ("medium", int(timesteps * 0.25)),
        ("hard", int(timesteps * 0.60)),
    ]

    print(f"Combined training: RND + DomainRand + Curriculum")
    print(f"  Total: {timesteps:,} steps, seed={seed}, rnd_coef={rnd_coef}")
    for diff, steps in phases:
        print(f"  Phase {diff}: {steps:,} steps")

    model = None
    for i, (difficulty, phase_steps) in enumerate(phases):
        phase_dir = out / difficulty
        phase_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f"Phase {i+1}/3: {difficulty} ({phase_steps:,} steps)")
        print(f"  Domain rand + RND (coef={rnd_coef})")
        print(f"{'='*50}")

        # Training env: domain rand + RND
        train_env = make_vec_env(
            lambda d=difficulty, s=seed, c=rnd_coef: _make_combined_env(d, s, c),
            n_envs=8,
        )
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=False, clip_reward=5.0,
        )

        # Eval env: clean (no RND, no domain rand) on hard
        eval_env = make_vec_env(
            lambda: _make_eval_env(difficulty="hard", seed=seed + 1000),
            n_envs=1,
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

    # Copy final hard model to top level
    import shutil
    shutil.copy(str(out / "hard" / "final_model.zip"), str(out / "final_model.zip"))
    shutil.copy(str(out / "hard" / "vec_normalize.pkl"), str(out / "vec_normalize.pkl"))
    print(f"\nCombined training complete. Final model: {out / 'final_model.zip'}")

    return out / "final_model.zip"


def evaluate_combined(
    model_path: str, n_episodes: int = 200, vec_normalize_path: str | None = None,
) -> dict[str, float]:
    """Evaluate on clean hard env (no wrappers)."""
    model = PPO.load(model_path)

    eval_venv = make_vec_env(
        lambda: _make_eval_env(difficulty="hard", seed=123), n_envs=1,
    )
    if vec_normalize_path is None:
        candidate = Path(model_path).parent.parent / "vec_normalize.pkl"
        if candidate.exists():
            vec_normalize_path = str(candidate)
    if vec_normalize_path and Path(vec_normalize_path).exists():
        eval_venv = VecNormalize.load(vec_normalize_path, eval_venv)
        eval_venv.training = False
        eval_venv.norm_reward = False

    successes, total_steps, total_reward = 0, 0, 0.0

    for ep in range(n_episodes):
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

    return {
        "method": "combined_rnd_domrand_curriculum",
        "success_rate": round(successes / n_episodes, 3),
        "avg_steps": round(total_steps / n_episodes, 1),
        "avg_reward": round(total_reward / n_episodes, 3),
        "n_episodes": n_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combined training: RND + DomainRand + Curriculum")
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--output", type=str, default="outputs/rl_combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rnd-coef", type=float, default=0.1)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-model", type=str, default=None)
    args = parser.parse_args()

    if args.eval:
        model_path = args.eval_model or f"{args.output}/hard/best_model/best_model"
        print(f"Evaluating {model_path} on hard (200 episodes)...")
        metrics = evaluate_combined(model_path)
        print(json.dumps(metrics, indent=2))
    else:
        train_combined(
            timesteps=args.timesteps,
            output_dir=args.output,
            seed=args.seed,
            rnd_coef=args.rnd_coef,
        )


if __name__ == "__main__":
    main()
