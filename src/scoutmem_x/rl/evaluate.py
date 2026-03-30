"""Multi-seed training + evaluation harness for ScoutMem-X RL experiments.

Trains N seeds, evaluates each, and reports mean +/- std across seeds.

Usage:
    python -m scoutmem_x.rl.evaluate --seeds 0,1000,2000,3000,4000
    python -m scoutmem_x.rl.evaluate --difficulty hard --timesteps 100000
    python -m scoutmem_x.rl.evaluate --eval-only --output outputs/rl_multiseed
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    seed: int
    success_rate: float
    avg_steps: float
    avg_reward: float
    model_dir: str
    train_time_s: float | None = None


@dataclass(frozen=True)
class Summary:
    n_seeds: int
    difficulty: str
    timesteps: int
    success_rate_mean: float
    success_rate_std: float
    avg_steps_mean: float
    avg_steps_std: float
    avg_reward_mean: float
    avg_reward_std: float
    per_seed: list[SeedResult]


# ---------------------------------------------------------------------------
# Env factory (re-used from train.py)
# ---------------------------------------------------------------------------

_FACTORIES = {"easy": ScoutMemEnv.easy, "medium": ScoutMemEnv.medium, "hard": ScoutMemEnv.hard}


def make_env(difficulty: str = "hard", seed: int = 0) -> ScoutMemEnv:
    env = _FACTORIES[difficulty]()
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_single_seed(
    seed: int, timesteps: int, difficulty: str, output_dir: Path
) -> Path:
    """Train PPO for one seed and return the output directory."""
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(lambda: make_env(difficulty=difficulty, seed=seed), n_envs=8)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_reward=5.0)

    eval_env = make_vec_env(lambda: make_env(difficulty=difficulty, seed=seed + 1000), n_envs=1)
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
        tensorboard_log=None,
        policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(seed_dir / "best_model"),
        log_path=str(seed_dir / "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=30,
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=False)
    model.save(str(seed_dir / "final_model"))
    train_env.save(str(seed_dir / "vec_normalize.pkl"))

    return seed_dir


def multi_seed_train(
    seeds: list[int],
    timesteps: int = 100_000,
    difficulty: str = "hard",
    output_dir: str = "outputs/rl_multiseed",
) -> list[Path]:
    """Train PPO for each seed sequentially. Returns list of per-seed output dirs."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] Training seed={seed}  "
              f"({difficulty}, {timesteps:,} steps) ...")
        t0 = time.time()
        d = _train_single_seed(seed, timesteps, difficulty, base)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s  ->  {d}")
        dirs.append(d)

    return dirs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval_single(
    model_dir: Path, difficulty: str, n_episodes: int = 100
) -> dict[str, float]:
    """Evaluate a single trained model and return metrics dict."""
    model_path = model_dir / "best_model" / "best_model"
    if not model_path.with_suffix(".zip").exists():
        model_path = model_dir / "final_model"

    model = PPO.load(str(model_path))

    eval_venv = make_vec_env(lambda: make_env(difficulty=difficulty, seed=123), n_envs=1)
    vnorm_path = model_dir / "vec_normalize.pkl"
    if vnorm_path.exists():
        eval_venv = VecNormalize.load(str(vnorm_path), eval_venv)
        eval_venv.training = False
        eval_venv.norm_reward = False

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

    return {
        "success_rate": successes / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_reward": total_reward / n_episodes,
    }


def multi_seed_eval(
    model_dirs: list[Path],
    difficulty: str = "hard",
    n_episodes: int = 100,
) -> list[SeedResult]:
    """Evaluate each trained model and return per-seed results."""
    results: list[SeedResult] = []

    for i, d in enumerate(model_dirs, 1):
        seed = int(d.name.split("_")[-1])
        print(f"[{i}/{len(model_dirs)}] Evaluating seed={seed}  "
              f"({n_episodes} episodes, {difficulty}) ...")
        metrics = _eval_single(d, difficulty, n_episodes)
        result = SeedResult(
            seed=seed,
            success_rate=metrics["success_rate"],
            avg_steps=metrics["avg_steps"],
            avg_reward=metrics["avg_reward"],
            model_dir=str(d),
        )
        results.append(result)
        print(f"  success={result.success_rate:.1%}  "
              f"steps={result.avg_steps:.1f}  "
              f"reward={result.avg_reward:.2f}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(
    results: list[SeedResult], difficulty: str = "hard", timesteps: int = 100_000
) -> Summary:
    """Compute mean +/- std across seeds for all metrics."""
    sr = np.array([r.success_rate for r in results])
    st = np.array([r.avg_steps for r in results])
    rw = np.array([r.avg_reward for r in results])

    return Summary(
        n_seeds=len(results),
        difficulty=difficulty,
        timesteps=timesteps,
        success_rate_mean=float(np.mean(sr)),
        success_rate_std=float(np.std(sr)),
        avg_steps_mean=float(np.mean(st)),
        avg_steps_std=float(np.std(st)),
        avg_reward_mean=float(np.mean(rw)),
        avg_reward_std=float(np.std(rw)),
        per_seed=results,
    )


# ---------------------------------------------------------------------------
# Pretty table
# ---------------------------------------------------------------------------


def _print_table(summary: Summary) -> None:
    w = 64
    print()
    print("=" * w)
    print(f"  Multi-Seed Results  |  {summary.difficulty}  "
          f"|  {summary.n_seeds} seeds  |  {summary.timesteps:,} steps")
    print("=" * w)

    header = f"{'Seed':>6}  {'Success':>10}  {'Avg Steps':>10}  {'Avg Reward':>12}"
    print(header)
    print("-" * w)
    for r in summary.per_seed:
        print(f"{r.seed:>6}  {r.success_rate:>9.1%}  "
              f"{r.avg_steps:>10.1f}  {r.avg_reward:>12.2f}")
    print("-" * w)
    print(f"{'MEAN':>6}  {summary.success_rate_mean:>9.1%}  "
          f"{summary.avg_steps_mean:>10.1f}  {summary.avg_reward_mean:>12.2f}")
    print(f"{'STD':>6}  {summary.success_rate_std:>9.1%}  "
          f"{summary.avg_steps_std:>10.1f}  {summary.avg_reward_std:>12.2f}")
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed RL training and evaluation for ScoutMem-X"
    )
    parser.add_argument(
        "--seeds", type=str, default="0,1000,2000,3000,4000",
        help="Comma-separated seeds (default: 0,1000,2000,3000,4000)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Training timesteps per seed (default: 100000)",
    )
    parser.add_argument(
        "--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"],
        help="Environment difficulty (default: hard)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/rl_multiseed",
        help="Base output directory (default: outputs/rl_multiseed)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; evaluate existing models in --output",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Episodes per seed during evaluation (default: 100)",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    base = Path(args.output)

    # -- Train (unless --eval-only) ------------------------------------------
    if args.eval_only:
        model_dirs = sorted(base.glob("seed_*"))
        if not model_dirs:
            print(f"No seed_* directories found in {base}. Train first.")
            return
        print(f"Eval-only mode: found {len(model_dirs)} seed dirs in {base}")
    else:
        model_dirs = multi_seed_train(
            seeds=seeds,
            timesteps=args.timesteps,
            difficulty=args.difficulty,
            output_dir=args.output,
        )

    # -- Evaluate -------------------------------------------------------------
    results = multi_seed_eval(
        model_dirs=model_dirs,
        difficulty=args.difficulty,
        n_episodes=args.eval_episodes,
    )

    # -- Summarize ------------------------------------------------------------
    summary = summarize(results, difficulty=args.difficulty, timesteps=args.timesteps)
    _print_table(summary)

    # -- Save -----------------------------------------------------------------
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / "summary.json"
    with open(out_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
