"""Compare ScoutMem-X (RL + memory) vs baselines.

This is the core experiment that proves the project's value.

Baselines:
1. Random exploration + vector DB (single-shot retrieval)
2. Random exploration + ScoutMem-X memory
3. Rule-based exploration + ScoutMem-X memory
4. RL-trained exploration + ScoutMem-X memory

Usage:
    python -m scoutmem_x.rl.compare
    python -m scoutmem_x.rl.compare --rl-model outputs/rl/final_model
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

import faiss

from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.rl.env import ScoutMemEnv


def run_vector_db_baseline(n_episodes: int = 100, seed: int = 42) -> dict[str, float]:
    """Vector DB baseline: store CLIP-like embeddings, retrieve by similarity.

    This simulates what a FAISS/ChromaDB system would do:
    - Each detection is stored as a single embedding
    - On search, return the most similar embedding
    - No confidence aggregation, no temporal decay, no multi-frame evidence
    - Just: store once, retrieve by similarity

    This is the baseline ScoutMem-X must beat.
    """
    env = ScoutMemEnv(grid_size=5, max_steps=25)
    successes, total_steps, total_reward = 0, 0, 0.0
    embed_dim = 16

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0

        # Vector DB: store embeddings as we explore
        index = faiss.IndexFlatL2(embed_dim)
        stored_labels: list[str] = []
        stored_positions: list[tuple[float, float]] = []
        visited_cells: set[tuple[int, int]] = {info["agent_pos"]}

        for step in range(env.max_steps):
            # Explore randomly
            action = _pick_exploration_action(
                info["agent_pos"], visited_cells, env.grid_size
            )

            obs_vec, reward, term, trunc, info = env.step(action)
            visited_cells.add(info["agent_pos"])
            ep_reward += reward

            # Store detections as embeddings in FAISS (simulated)
            # Each detection becomes ONE embedding — no aggregation
            if env._memory:
                for node in env._memory.nodes:
                    if node.category not in stored_labels:
                        # Create a simulated embedding from the label + position
                        embed = _make_embedding(
                            node.category, node.position_estimate, embed_dim
                        )
                        index.add(embed.reshape(1, -1).astype(np.float32))
                        stored_labels.append(node.category)
                        stored_positions.append(
                            (node.position_estimate[0], node.position_estimate[1])
                            if node.position_estimate
                            else (0, 0)
                        )

            # After exploring enough, try to find the target
            if step >= env.max_steps - 1 or len(visited_cells) >= env.grid_size ** 2 * 0.6:
                if index.ntotal > 0 and env._target_label:
                    # Query FAISS: find most similar embedding to target
                    query_embed = _make_query_embedding(
                        env._target_label, embed_dim
                    )
                    _, indices = index.search(
                        query_embed.reshape(1, -1).astype(np.float32), 1
                    )
                    best_idx = indices[0][0]
                    if 0 <= best_idx < len(stored_positions):
                        retrieved_pos = stored_positions[best_idx]
                        target_pos = tuple(env._target_pos)
                        dist = np.sqrt(
                            (retrieved_pos[0] - target_pos[0]) ** 2
                            + (retrieved_pos[1] - target_pos[1]) ** 2
                        )
                        if dist <= 1.5:
                            successes += 1
                            ep_reward += 10.0
                        else:
                            ep_reward -= 5.0
                total_steps += step + 1
                break

            if term or trunc:
                total_steps += step + 1
                break

        total_reward += ep_reward

    return {
        "baseline": "vector_db_faiss",
        "success_rate": round(successes / n_episodes, 3),
        "avg_steps": round(total_steps / n_episodes, 1),
        "avg_reward": round(total_reward / n_episodes, 2),
    }


def _make_embedding(
    label: str, position: tuple[float, float, float] | None, dim: int
) -> np.ndarray:
    """Simulate a CLIP-like embedding from label + position.

    In a real system this would be CLIP(image_crop). Here we simulate it
    with a deterministic hash + noise to model the fact that embeddings
    of similar objects are close but not identical.
    """
    rng = np.random.RandomState(hash(label) % (2**31))
    base = rng.randn(dim).astype(np.float32)
    # Add positional info
    if position:
        base[0] += position[0] * 0.1
        base[1] += position[1] * 0.1
    # Add noise (simulates embedding variability)
    noise = np.random.randn(dim).astype(np.float32) * 0.15
    return base + noise


def _make_query_embedding(label: str, dim: int) -> np.ndarray:
    """Create query embedding (text-to-embedding, no noise)."""
    rng = np.random.RandomState(hash(label) % (2**31))
    return rng.randn(dim).astype(np.float32)


def run_random_baseline(n_episodes: int = 100, seed: int = 42) -> dict[str, float]:
    """Random exploration — no intelligence, just moves randomly then stops."""
    env = ScoutMemEnv(grid_size=5, max_steps=25)
    successes, total_steps, total_reward = 0, 0, 0.0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for step in range(env.max_steps):
            action = random.choice([0, 1, 2, 3])  # random movement only
            if step == env.max_steps - 1:
                action = 4  # force stop at end
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc:
                if reward > 0:
                    successes += 1
                total_steps += step + 1
                break
        total_reward += ep_reward

    return {
        "baseline": "random_exploration",
        "success_rate": round(successes / n_episodes, 3),
        "avg_steps": round(total_steps / n_episodes, 1),
        "avg_reward": round(total_reward / n_episodes, 2),
    }


def run_rule_based(n_episodes: int = 100, seed: int = 42) -> dict[str, float]:
    """Rule-based exploration + ScoutMem-X memory (active evidence policy)."""
    env = ScoutMemEnv(grid_size=5, max_steps=25)
    successes, total_steps, total_reward = 0, 0, 0.0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        visited_cells: set[tuple[int, int]] = {info["agent_pos"]}

        for step in range(env.max_steps):
            conf = info["target_confidence"]

            # Rule-based policy (5 actions: 0-3 move, 4 stop):
            # 1. If confidence >= 0.5, stop (we're sure enough)
            if conf >= 0.5:
                action = 4  # STOP
            # 2. Otherwise, move toward unexplored areas
            else:
                action = _pick_exploration_action(
                    info["agent_pos"], visited_cells, env.grid_size
                )

            obs, reward, term, trunc, info = env.step(action)
            visited_cells.add(info["agent_pos"])
            ep_reward += reward
            if term or trunc:
                if reward > 0:
                    successes += 1
                total_steps += step + 1
                break
        total_reward += ep_reward

    return {
        "baseline": "rule_based_scoutmem",
        "success_rate": round(successes / n_episodes, 3),
        "avg_steps": round(total_steps / n_episodes, 1),
        "avg_reward": round(total_reward / n_episodes, 2),
    }


def run_rl_policy(
    model_path: str, n_episodes: int = 100, seed: int = 42,
    vec_normalize_path: str | None = None,
) -> dict[str, float]:
    """RL-trained exploration + ScoutMem-X memory."""
    from pathlib import Path

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    model = PPO.load(model_path)

    # Use VecNormalize to match training conditions
    eval_venv = make_vec_env(
        lambda: ScoutMemEnv(grid_size=5, max_steps=25), n_envs=1,
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
        "baseline": "rl_trained_scoutmem",
        "success_rate": round(successes / n_episodes, 3),
        "avg_steps": round(total_steps / n_episodes, 1),
        "avg_reward": round(total_reward / n_episodes, 2),
    }


def _pick_exploration_action(
    agent_pos: tuple[int, int], visited: set[tuple[int, int]], grid_size: int
) -> int:
    """Pick movement toward the nearest unvisited cell."""
    ax, ay = agent_pos
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

    best_action = 0
    best_unvisited_dist = float("inf")

    for action_idx, (dx, dy) in enumerate(directions):
        nx, ny = ax + dx, ay + dy
        if not (0 <= nx < grid_size and 0 <= ny < grid_size):
            continue
        if (nx, ny) not in visited:
            return action_idx  # go to first unvisited neighbor

        # If all neighbors visited, move toward the farthest unvisited cell
        for ux in range(grid_size):
            for uy in range(grid_size):
                if (ux, uy) not in visited:
                    dist = abs(nx - ux) + abs(ny - uy)
                    if dist < best_unvisited_dist:
                        best_unvisited_dist = dist
                        best_action = action_idx

    return best_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ScoutMem-X baselines")
    parser.add_argument("--rl-model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("ScoutMem-X Comparison Experiment")
    print("=" * 60)
    print()

    results = []

    # Baseline 0: Vector DB (FAISS)
    print("Running: Vector DB (FAISS) — single-shot retrieval...")
    r0 = run_vector_db_baseline(n_episodes=args.episodes)
    results.append(r0)
    print(f"  Success: {r0['success_rate']:.1%} | Steps: {r0['avg_steps']:.1f}")

    # Baseline 1: Random exploration + ScoutMem-X
    print("Running: Random exploration + ScoutMem-X memory...")
    r1 = run_random_baseline(n_episodes=args.episodes)
    results.append(r1)
    print(f"  Success: {r1['success_rate']:.1%} | Steps: {r1['avg_steps']:.1f}")

    # Baseline 2: Rule-based + ScoutMem-X
    print("Running: Rule-based + ScoutMem-X memory...")
    r2 = run_rule_based(n_episodes=args.episodes)
    results.append(r2)
    print(f"  Success: {r2['success_rate']:.1%} | Steps: {r2['avg_steps']:.1f}")

    # Baseline 3: RL-trained + ScoutMem-X
    if args.rl_model:
        print(f"Running: RL-trained + ScoutMem-X ({args.rl_model})...")
        r3 = run_rl_policy(args.rl_model, n_episodes=args.episodes)
        results.append(r3)
        print(f"  Success: {r3['success_rate']:.1%} | Steps: {r3['avg_steps']:.1f}")

    print()
    print("=" * 60)
    print(f"{'Baseline':<30} {'Success':>10} {'Avg Steps':>10} {'Avg Reward':>12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['baseline']:<30} {r['success_rate']:>9.1%} {r['avg_steps']:>10.1f} "
            f"{r['avg_reward']:>12.2f}"
        )
    print("=" * 60)

    # Save results
    out = Path("outputs/comparison")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out / 'results.json'}")


if __name__ == "__main__":
    main()
