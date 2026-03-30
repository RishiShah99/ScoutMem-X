"""Ablation study runner for ScoutMem-X.

Tests which components contribute to performance by training/evaluating
PPO under 6 conditions, each with 3 seeds.

Conditions:
    full           — full model (frame stacking + belief features + conf reward + memory decay)
    no_frame_stack — frame_stack=1 instead of 4 (16-dim obs instead of 64)
    no_belief      — zero out belief features (dims 10-15 of each 16-dim frame)
    no_conf_reward — remove confidence progress reward (only step cost + stop rewards)
    no_decay       — decay_rate=0.0 (memory never decays)
    random_policy  — random actions (no RL, just ScoutMem memory)

Usage:
    python -m scoutmem_x.rl.ablation
    python -m scoutmem_x.rl.ablation --conditions full,no_belief,random_policy
    python -m scoutmem_x.rl.ablation --eval-only --output outputs/ablation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv

ALL_CONDITIONS = [
    "full", "no_frame_stack", "no_belief", "no_conf_reward", "no_decay", "random_policy",
]
SEEDS = [0, 1000, 2000]
N_EVAL_EPISODES = 100


# ── Wrappers ───────────────────────────────────────────────────


class NoBeliefWrapper(gym.ObservationWrapper):
    """Zero out belief features (dims 10-15) of each 16-dim frame."""

    def observation(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        frame_dim = 16
        n_frames = len(out) // frame_dim
        for i in range(n_frames):
            out[i * frame_dim + 10 : i * frame_dim + 16] = 0.0
        return out


class NoConfRewardWrapper(gym.RewardWrapper):
    """Remove confidence progress reward — keep only step cost and stop rewards.

    Step cost is ~-0.04 and stop rewards are +1.0 (correct) or -1.0 (wrong).
    Confidence bonuses are small positive increments (0.5 * delta, typically <0.1).
    Simplest approach: clamp non-terminal rewards to the step cost.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._step_cost = -1.0 / env.max_steps  # type: ignore[attr-defined]

    def reward(self, reward: float) -> float:
        # Terminal rewards (stop or truncation) are large magnitude — pass through
        if abs(reward) >= 0.4:
            return reward
        # Non-terminal: strip confidence bonus, keep step cost + exploration bonus
        return min(reward, self._step_cost + 0.02)


# ── Environment factories ──────────────────────────────────────


def _make_env(condition: str, seed: int) -> gym.Env:
    """Create a single env for the given ablation condition."""
    if condition == "no_frame_stack":
        env = ScoutMemEnv.hard(frame_stack=1)
    elif condition == "no_decay":
        env = ScoutMemEnv.hard(decay_rate=0.0)
    else:
        env = ScoutMemEnv.hard()

    if condition == "no_belief":
        env = NoBeliefWrapper(env)
    elif condition == "no_conf_reward":
        env = NoConfRewardWrapper(env)

    env.reset(seed=seed)
    return env


# ── Training ───────────────────────────────────────────────────


def train_condition(
    condition: str,
    seed: int,
    timesteps: int,
    output_dir: Path,
) -> Path:
    """Train PPO for one (condition, seed) pair. Returns model path."""
    run_dir = output_dir / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training {condition} seed={seed} for {timesteps:,} steps ...")

    train_env = make_vec_env(lambda: _make_env(condition, seed), n_envs=8)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_reward=5.0)

    eval_env = make_vec_env(lambda: _make_env(condition, seed + 500), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
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
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    model.save(str(run_dir / "final_model"))
    train_env.save(str(run_dir / "vec_normalize.pkl"))

    best_path = run_dir / "best_model" / "best_model.zip"
    return best_path if best_path.exists() else run_dir / "final_model.zip"


# ── Evaluation ─────────────────────────────────────────────────


def evaluate_random(seed: int, n_episodes: int = N_EVAL_EPISODES) -> dict[str, float]:
    """Evaluate random policy (no RL) with ScoutMem memory."""
    env = ScoutMemEnv.hard()
    successes, total_steps, total_reward = 0, 0, 0.0

    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(env.max_steps):
            action = random.choice([0, 1, 2, 3])
            if step == env.max_steps - 1:
                action = 4  # force stop at end
            _, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if term or trunc:
                if reward > 0:
                    successes += 1
                total_steps += step + 1
                break
        total_reward += ep_reward

    return {
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def evaluate_trained(
    condition: str,
    model_path: Path,
    vec_norm_path: Path | None,
    n_episodes: int = N_EVAL_EPISODES,
) -> dict[str, float]:
    """Evaluate a trained PPO model for the given condition."""
    model = PPO.load(str(model_path))

    eval_venv = make_vec_env(lambda: _make_env(condition, seed=9999), n_envs=1)
    if vec_norm_path and vec_norm_path.exists():
        eval_venv = VecNormalize.load(str(vec_norm_path), eval_venv)
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
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


# ── Orchestration ──────────────────────────────────────────────


def run_ablation(
    conditions: list[str],
    timesteps: int,
    output_dir: str,
    eval_only: bool = False,
) -> dict[str, Any]:
    """Run the full ablation study. Returns aggregated results dict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for cond in conditions:
        print(f"\n{'='*50}")
        print(f"Condition: {cond}")
        print(f"{'='*50}")

        seed_metrics: list[dict[str, float]] = []

        for seed in SEEDS:
            if cond == "random_policy":
                metrics = evaluate_random(seed=seed)
            else:
                run_dir = out / cond / f"seed_{seed}"
                model_path = run_dir / "best_model" / "best_model.zip"
                if not model_path.exists():
                    model_path = run_dir / "final_model.zip"
                vec_norm_path = run_dir / "vec_normalize.pkl"

                if not eval_only:
                    train_condition(cond, seed, timesteps, out)

                if not model_path.exists():
                    print(f"  [SKIP] No model found at {model_path}")
                    continue

                metrics = evaluate_trained(cond, model_path, vec_norm_path)

            seed_metrics.append(metrics)
            print(
                f"  seed={seed}: success={metrics['success_rate']:.1%} "
                f"steps={metrics['avg_steps']:.1f} reward={metrics['avg_reward']:.2f}"
            )

        if not seed_metrics:
            continue

        # Aggregate across seeds
        agg: dict[str, Any] = {}
        for key in ["success_rate", "avg_steps", "avg_reward"]:
            vals = [m[key] for m in seed_metrics]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        agg["per_seed"] = seed_metrics
        results[cond] = agg

    # Save results
    results_path = out / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    _print_table(results)
    return results


def _print_table(results: dict[str, Any]) -> None:
    """Print a formatted summary table."""
    print()
    print("=" * 78)
    print(f"{'Condition':<20} {'Success Rate':>16} {'Avg Steps':>16} {'Avg Reward':>16}")
    print("-" * 78)
    for cond, agg in results.items():
        sr = agg["success_rate"]
        st = agg["avg_steps"]
        rw = agg["avg_reward"]
        print(
            f"{cond:<20} "
            f"{sr['mean']:>6.1%} +/- {sr['std']:.1%}  "
            f"{st['mean']:>5.1f} +/- {st['std']:>4.1f}  "
            f"{rw['mean']:>5.2f} +/- {rw['std']:>4.2f}"
        )
    print("=" * 78)


# ── CLI ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ScoutMem-X ablation study")
    parser.add_argument(
        "--conditions", type=str, default=",".join(ALL_CONDITIONS),
        help="Comma-separated conditions to run (default: all)",
    )
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--output", type=str, default="outputs/ablation")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate existing models")
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    invalid = [c for c in conditions if c not in ALL_CONDITIONS]
    if invalid:
        parser.error(f"Unknown conditions: {invalid}. Valid: {ALL_CONDITIONS}")

    run_ablation(
        conditions=conditions,
        timesteps=args.timesteps,
        output_dir=args.output,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
