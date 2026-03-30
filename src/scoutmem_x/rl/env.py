"""Gymnasium environment for training RL exploration policies with ScoutMem-X.

The agent navigates a grid world with noisy perception, building structured
memory through Bayesian confidence aggregation. Frame stacking gives the
policy temporal history to handle partial observability (POMDP).
"""

from __future__ import annotations

import collections
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
    """Gymnasium environment for embodied object search with ScoutMem-X.

    Key design decisions:
    - 16-dim per-frame observation with belief features (quadrant coverage,
      steps since progress, max confidence ever seen)
    - Frame stacking (default 4) to give the policy temporal history
    - Bounded reward in [-1.5, +1.0] for stable value learning
    - Difficulty factory methods: easy/medium/hard
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
        frame_stack: int = 4,
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
        self.frame_stack = frame_stack
        self.render_mode = render_mode

        # 5 actions: 4 movement + stop
        self.action_space = spaces.Discrete(5)

        # 16 features per frame x frame_stack frames
        self._frame_dim = 16
        obs_dim = self._frame_dim * frame_stack
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Will be initialized in reset()
        self._agent_pos = np.array([0, 0])
        self._objects: list[dict[str, Any]] = []
        self._target_label = ""
        self._target_pos = np.array([0, 0])
        self._memory: MemorySnapshot | None = None
        self._step_count = 0
        self._visited: set[tuple[int, int]] = set()
        self._prev_target_conf = 0.0
        self._steps_since_conf_gain = 0
        self._max_conf_ever = 0.0
        self._obs_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=frame_stack,
        )

    # ── difficulty factories ────────────────────────────────────

    @classmethod
    def easy(cls, **kw: Any) -> "ScoutMemEnv":
        """3x3 grid, 3 objects, 0 distractors, low noise."""
        return cls(
            grid_size=3, n_objects=3, n_distractors=0,
            view_range=2.0, dropout_rate=0.05, noise_std=0.03,
            max_steps=15, **kw,
        )

    @classmethod
    def medium(cls, **kw: Any) -> "ScoutMemEnv":
        """4x4 grid, 5 objects, 1 distractor."""
        return cls(
            grid_size=4, n_objects=5, n_distractors=1,
            view_range=2.0, dropout_rate=0.08, noise_std=0.05,
            max_steps=20, **kw,
        )

    @classmethod
    def hard(cls, **kw: Any) -> "ScoutMemEnv":
        """5x5 grid, 6 objects, 2 distractors (default)."""
        return cls(
            grid_size=5, n_objects=6, n_distractors=2,
            view_range=2.0, dropout_rate=0.10, noise_std=0.06,
            max_steps=25, **kw,
        )

    # ── core API ────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
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
            if i >= self.n_objects:
                label = labels[0]  # distractors share target label
            self._objects.append({
                "label": label,
                "pos": np.array(pos),
                "base_visibility": 0.6 + self.np_random.random() * 0.35,
                "is_target": i == 0,
            })

        self._target_label = self._objects[0]["label"]
        self._target_pos = self._objects[0]["pos"]
        self._memory = None
        self._step_count = 0
        self._visited = {tuple(self._agent_pos)}
        self._prev_target_conf = 0.0
        self._steps_since_conf_gain = 0
        self._max_conf_ever = 0.0

        self._perceive()

        # Fill frame buffer with initial observation
        frame = self._build_frame()
        self._obs_buffer.clear()
        for _ in range(self.frame_stack):
            self._obs_buffer.append(frame)

        return self._stacked_obs(), self._get_info()

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        # ── reward v6: bounded [-1.5, +1.0] ──
        reward = -1.0 / self.max_steps  # step cost ~-0.04

        if action <= 3:  # move
            direction = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_pos = self._agent_pos + np.array(direction)
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                self._agent_pos = new_pos
            new_cell = tuple(self._agent_pos)
            if new_cell not in self._visited:
                reward += 0.02  # tiny exploration bonus
                self._visited.add(new_cell)

        elif action == 4:  # stop
            best = retrieve_best_node(
                self._memory, self._target_label,
            ) if self._memory else None

            if best is not None and best.position_estimate is not None:
                est = np.array([best.position_estimate[0], best.position_estimate[1]])
                dist = float(np.linalg.norm(est - self._target_pos))
                if dist <= 1.5 and best.confidence >= 0.5:
                    reward = 1.0  # correct
                else:
                    reward = -1.0  # wrong
            else:
                reward = -1.0

            self._obs_buffer.append(self._build_frame())
            return self._stacked_obs(), reward, True, False, self._get_info()

        # Perceive after moving
        self._perceive()

        # Confidence progress reward
        best = retrieve_best_node(
            self._memory, self._target_label,
        ) if self._memory else None
        current_conf = best.confidence if best else 0.0

        conf_delta = current_conf - self._prev_target_conf
        if conf_delta > 0.01:
            reward += 0.5 * conf_delta
            self._steps_since_conf_gain = 0
        else:
            self._steps_since_conf_gain += 1

        self._max_conf_ever = max(self._max_conf_ever, current_conf)
        self._prev_target_conf = current_conf

        # Timeout
        truncated = self._step_count >= self.max_steps
        if truncated:
            reward = -0.5

        self._obs_buffer.append(self._build_frame())
        return self._stacked_obs(), reward, False, truncated, self._get_info()

    # ── perception ──────────────────────────────────────────────

    def _perceive(self) -> None:
        detections: list[Detection] = []
        for obj in self._objects:
            dist = float(np.linalg.norm(self._agent_pos - obj["pos"]))
            if dist > self.view_range:
                continue
            if random.random() < self.dropout_rate:
                continue

            raw_conf = obj["base_visibility"] * (1.0 - (dist / self.view_range) ** 0.6)
            noisy_conf = raw_conf + random.gauss(0, self.noise_std)
            noisy_conf = max(0.05, min(0.95, noisy_conf))

            region = f"zone_{int(obj['pos'][0])}_{int(obj['pos'][1])}"
            detections.append(Detection(
                label=obj["label"],
                score=round(noisy_conf, 3),
                region=(0, 0, 64, 64),
                metadata={
                    "query": "", "source": "spatial_sim",
                    "region": region, "target_label": self._target_label,
                },
            ))

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

    # ── observation ─────────────────────────────────────────────

    def _build_frame(self) -> np.ndarray:
        """Build a 16-dim frame with belief features."""
        gs = float(self.grid_size)

        # Agent position (2)
        ax = self._agent_pos[0] / gs
        ay = self._agent_pos[1] / gs

        # Target confidence and direction (3)
        best = retrieve_best_node(
            self._memory, self._target_label,
        ) if self._memory else None
        target_conf = best.confidence if best else 0.0
        target_dx, target_dy = 0.5, 0.5
        if best and best.position_estimate:
            est = np.array([best.position_estimate[0], best.position_estimate[1]])
            diff = est - self._agent_pos
            target_dx = float(np.clip(diff[0] / gs, -1, 1) * 0.5 + 0.5)
            target_dy = float(np.clip(diff[1] / gs, -1, 1) * 0.5 + 0.5)

        # Coverage (1)
        coverage = len(self._visited) / (gs * gs)

        # Time remaining (1)
        time_left = 1.0 - (self._step_count / self.max_steps)

        # Candidates count (1)
        n_candidates = 0
        if self._memory:
            n_candidates = sum(
                1 for n in self._memory.nodes if n.category == self._target_label
            )
        n_cand_norm = min(n_candidates / 5.0, 1.0)

        # Direction to nearest unvisited (2)
        unvis_dx, unvis_dy = 0.5, 0.5
        best_dist = float("inf")
        for ux in range(self.grid_size):
            for uy in range(self.grid_size):
                if (ux, uy) not in self._visited:
                    d = abs(ux - self._agent_pos[0]) + abs(uy - self._agent_pos[1])
                    if d < best_dist:
                        best_dist = d
                        diff = np.array([ux - self._agent_pos[0], uy - self._agent_pos[1]])
                        unvis_dx = float(np.clip(diff[0] / gs, -1, 1) * 0.5 + 0.5)
                        unvis_dy = float(np.clip(diff[1] / gs, -1, 1) * 0.5 + 0.5)

        # ── 6 NEW belief features ──

        # Quadrant coverage (4): what fraction of each quadrant has been visited
        half = self.grid_size / 2.0
        quads = [0.0, 0.0, 0.0, 0.0]  # NW, NE, SW, SE
        quad_totals = [0, 0, 0, 0]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                qi = (0 if x < half else 1) + (0 if y < half else 2)
                quad_totals[qi] += 1
                if (x, y) in self._visited:
                    quads[qi] += 1
        for qi in range(4):
            if quad_totals[qi] > 0:
                quads[qi] /= quad_totals[qi]

        # Steps since confidence gain (1)
        stale = min(self._steps_since_conf_gain / self.max_steps, 1.0)

        # Max confidence ever seen (1)
        max_conf = self._max_conf_ever

        return np.array([
            ax, ay,                          # 0-1: position
            target_conf,                     # 2: current best confidence
            target_dx, target_dy,            # 3-4: direction to best candidate
            coverage,                        # 5: exploration progress
            time_left,                       # 6: urgency
            n_cand_norm,                     # 7: how many candidates found
            unvis_dx, unvis_dy,              # 8-9: where to explore next
            quads[0], quads[1],              # 10-11: quadrant NW, NE coverage
            quads[2], quads[3],              # 12-13: quadrant SW, SE coverage
            stale,                           # 14: is exploring still productive?
            max_conf,                        # 15: best confidence ever (prevents forgetting)
        ], dtype=np.float32)

    def _stacked_obs(self) -> np.ndarray:
        """Return concatenated frame stack."""
        return np.concatenate(list(self._obs_buffer))

    def _get_info(self) -> dict[str, Any]:
        best = retrieve_best_node(
            self._memory, self._target_label,
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
                elif any(np.array_equal(o["pos"], [x, y]) and o["is_target"] for o in self._objects):
                    row += " T "
                elif any(np.array_equal(o["pos"], [x, y]) for o in self._objects):
                    row += " o "
                elif (x, y) in self._visited:
                    row += " . "
                else:
                    row += " # "
            lines.append(row)
        best = retrieve_best_node(self._memory, self._target_label) if self._memory else None
        lines.append(f"Step: {self._step_count}/{self.max_steps}  Conf: {best.confidence:.2f}" if best else "Step: ?")
        return "\n".join(lines)
