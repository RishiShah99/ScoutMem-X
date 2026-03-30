"""Gymnasium environment for training RL exploration policies with ScoutMem-X.

The agent navigates a grid world with noisy perception, building structured
memory through Bayesian confidence aggregation. The RL policy learns WHERE
to explore to find a target object efficiently.
"""

from __future__ import annotations

import math
import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.memory.update import build_memory_snapshot
from scoutmem_x.perception.adapters import Detection


class ScoutMemEnv(gym.Env):
    """Gymnasium environment for embodied object search with ScoutMem-X memory.

    Action space (5 discrete):
        0-3: move (up/down/left/right)
        4:   stop and declare found

    Observation space (10-dim, all normalized 0-1):
        0-1: agent position (x, y)
        2:   best target confidence in memory
        3-4: direction to best target candidate (normalized)
        5:   exploration coverage (fraction of grid visited)
        6:   steps remaining (fraction)
        7:   number of target candidates in memory (normalized)
        8-9: direction to nearest unvisited cell

    Reward (deliberately simple and bounded):
        -0.1  per step
        +1.0  correct stop (target within range, confidence >= 0.5)
        -1.0  wrong stop
        +0.3  confidence gain on target (scaled by delta)
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 5,
        n_objects: int = 6,
        n_distractors: int = 2,
        view_range: float = 2.0,
        dropout_rate: float = 0.10,
        noise_std: float = 0.06,
        decay_rate: float = 0.01,
        max_steps: int = 25,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.n_objects = n_objects
        self.n_distractors = n_distractors
        self.view_range = view_range
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.decay_rate = decay_rate
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)  # 4 moves + stop

        # 10-dimensional observation, all in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        self._agent_pos = np.array([0, 0])
        self._objects: list[dict[str, Any]] = []
        self._target_label = ""
        self._target_pos = np.array([0, 0])
        self._memory: MemorySnapshot | None = None
        self._step_count = 0
        self._visited: set[tuple[int, int]] = set()
        self._prev_target_conf = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._agent_pos = np.array([
            self.np_random.integers(0, self.grid_size),
            self.np_random.integers(0, self.grid_size),
        ])

        self._objects = []
        positions_used: set[tuple[int, int]] = set()
        labels = [
            "mug", "book", "phone", "key", "lamp", "plant",
            "remote", "bottle", "clock", "pen", "vase", "laptop",
        ]

        for i in range(self.n_objects + self.n_distractors):
            while True:
                pos = (
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size),
                )
                if pos not in positions_used:
                    positions_used.add(pos)
                    break

            label = labels[i % len(labels)]
            is_target = i == 0
            if i >= self.n_objects:
                label = labels[0]  # distractors share target label

            self._objects.append({
                "label": label,
                "pos": np.array(pos),
                "base_visibility": 0.6 + self.np_random.random() * 0.35,
                "is_target": is_target,
            })

        self._target_label = self._objects[0]["label"]
        self._target_pos = self._objects[0]["pos"]
        self._memory = None
        self._step_count = 0
        self._visited = {tuple(self._agent_pos)}
        self._prev_target_conf = 0.0

        self._perceive()
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        reward = 0.0

        if action <= 3:  # move
            direction = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_pos = self._agent_pos + np.array(direction)
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                self._agent_pos = new_pos
            self._visited.add(tuple(self._agent_pos))

        elif action == 4:  # stop
            best = retrieve_best_node(
                self._memory, self._target_label
            ) if self._memory else None

            if best is not None and best.position_estimate is not None:
                est = np.array([best.position_estimate[0], best.position_estimate[1]])
                dist = float(np.linalg.norm(est - self._target_pos))
                if dist <= 1.5 and best.confidence >= 0.5:
                    reward = 10.0  # correct find — big reward
                else:
                    reward = -10.0  # wrong — big penalty, must hurt more than timeout
            else:
                reward = -10.0

            return self._get_obs(), reward, True, False, self._get_info()

        # Perceive after moving
        self._perceive()

        # Reward confidence gains on target (main learning signal)
        best = retrieve_best_node(
            self._memory, self._target_label
        ) if self._memory else None
        current_conf = best.confidence if best else 0.0
        conf_delta = current_conf - self._prev_target_conf
        if conf_delta > 0:
            reward += 1.0 * conf_delta  # reward getting closer to finding target
        self._prev_target_conf = current_conf

        truncated = self._step_count >= self.max_steps
        # Timeout: mild penalty, much less than wrong-STOP
        if truncated:
            reward = -2.0

        return self._get_obs(), reward, False, truncated, self._get_info()

    def _perceive(self, boost: bool = False) -> None:
        detections: list[Detection] = []

        for obj in self._objects:
            dist = float(np.linalg.norm(self._agent_pos - obj["pos"]))
            if dist > self.view_range:
                continue

            drop = self.dropout_rate * (0.3 if boost else 1.0)
            if random.random() < drop:
                continue

            raw_conf = obj["base_visibility"] * (1.0 - (dist / self.view_range) ** 0.6)
            if boost:
                raw_conf *= 1.3

            noisy_conf = raw_conf + random.gauss(0, self.noise_std)
            noisy_conf = max(0.05, min(0.95, noisy_conf))

            region = f"zone_{int(obj['pos'][0])}_{int(obj['pos'][1])}"
            detections.append(
                Detection(
                    label=obj["label"],
                    score=round(noisy_conf, 3),
                    region=(0, 0, 64, 64),
                    metadata={
                        "query": "",
                        "source": "spatial_sim",
                        "region": region,
                        "target_label": self._target_label,
                    },
                )
            )

        obs = Observation(
            frame_id=f"rl-step-{self._step_count}",
            step_index=self._step_count,
            pose=(float(self._agent_pos[0]), float(self._agent_pos[1]), 0.0),
            heading_radians=0.0,
            image_size=(64, 64),
        )

        self._memory = build_memory_snapshot(
            obs, detections, self._target_label, self._memory,
            decay_rate=self.decay_rate,
        )

    def _get_obs(self) -> np.ndarray:
        gs = float(self.grid_size)

        # Agent position normalized
        ax = self._agent_pos[0] / gs
        ay = self._agent_pos[1] / gs

        # Target confidence and direction
        best = retrieve_best_node(
            self._memory, self._target_label
        ) if self._memory else None
        target_conf = best.confidence if best else 0.0

        target_dx, target_dy = 0.0, 0.0
        if best and best.position_estimate:
            est = np.array([best.position_estimate[0], best.position_estimate[1]])
            diff = est - self._agent_pos
            norm = max(np.linalg.norm(diff), 0.01)
            target_dx = float(np.clip(diff[0] / gs, -1, 1) * 0.5 + 0.5)
            target_dy = float(np.clip(diff[1] / gs, -1, 1) * 0.5 + 0.5)

        # Coverage
        coverage = len(self._visited) / (gs * gs)

        # Time remaining
        time_left = 1.0 - (self._step_count / self.max_steps)

        # Number of target candidates in memory
        n_candidates = 0
        if self._memory:
            n_candidates = sum(
                1 for n in self._memory.nodes if n.category == self._target_label
            )
        n_candidates_norm = min(n_candidates / 5.0, 1.0)

        # Direction to nearest unvisited cell
        unvisited_dx, unvisited_dy = 0.5, 0.5
        best_dist = float("inf")
        for ux in range(self.grid_size):
            for uy in range(self.grid_size):
                if (ux, uy) not in self._visited:
                    d = abs(ux - self._agent_pos[0]) + abs(uy - self._agent_pos[1])
                    if d < best_dist:
                        best_dist = d
                        diff = np.array([ux - self._agent_pos[0], uy - self._agent_pos[1]])
                        norm = max(np.linalg.norm(diff), 0.01)
                        unvisited_dx = float(np.clip(diff[0] / gs, -1, 1) * 0.5 + 0.5)
                        unvisited_dy = float(np.clip(diff[1] / gs, -1, 1) * 0.5 + 0.5)

        return np.array([
            ax, ay,
            target_conf,
            target_dx, target_dy,
            coverage,
            time_left,
            n_candidates_norm,
            unvisited_dx, unvisited_dy,
        ], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        best = retrieve_best_node(
            self._memory, self._target_label
        ) if self._memory else None
        return {
            "step": self._step_count,
            "agent_pos": tuple(self._agent_pos),
            "target_pos": tuple(self._target_pos),
            "target_label": self._target_label,
            "target_confidence": best.confidence if best else 0.0,
            "memory_nodes": len(self._memory.nodes) if self._memory else 0,
            "coverage": len(self._visited) / (self.grid_size ** 2),
            "evidence_sufficiency": (
                self._memory.evidence_sufficiency_score if self._memory else 0.0
            ),
        }

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        lines = []
        for y in range(self.grid_size):
            row = ""
            for x in range(self.grid_size):
                if np.array_equal(self._agent_pos, [x, y]):
                    row += " A "
                elif any(
                    np.array_equal(o["pos"], [x, y]) and o["is_target"]
                    for o in self._objects
                ):
                    row += " T "
                elif any(np.array_equal(o["pos"], [x, y]) for o in self._objects):
                    row += " o "
                elif (x, y) in self._visited:
                    row += " . "
                else:
                    row += " # "
            lines.append(row)
        best = retrieve_best_node(
            self._memory, self._target_label
        ) if self._memory else None
        lines.append(f"Step: {self._step_count}/{self.max_steps}")
        lines.append(
            f"Target: {self._target_label} conf={best.confidence:.2f}"
            if best else "Target: ?"
        )
        return "\n".join(lines)
