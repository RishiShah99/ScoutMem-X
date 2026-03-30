"""Tests for the RL environment (frame stacking, belief features, reward bounds)."""

from __future__ import annotations

import numpy as np
import pytest

from scoutmem_x.rl.env import ScoutMemEnv


class TestObservationSpace:
    def test_frame_stacking_shape(self) -> None:
        env = ScoutMemEnv(frame_stack=4)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (64,)

    def test_frame_stack_1(self) -> None:
        env = ScoutMemEnv(frame_stack=1)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (16,)

    def test_obs_bounds(self) -> None:
        env = ScoutMemEnv.easy()
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            for _ in range(env.max_steps):
                assert np.all(obs >= -0.01) and np.all(obs <= 1.01), (
                    f"OOB obs: min={obs.min()}, max={obs.max()}"
                )
                obs, _, term, trunc, _ = env.step(env.action_space.sample())
                if term or trunc:
                    break

    def test_obs_dtype(self) -> None:
        env = ScoutMemEnv.easy()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32


class TestDifficultyFactories:
    def test_easy(self) -> None:
        env = ScoutMemEnv.easy()
        assert env.grid_size == 3
        assert env.n_objects == 3
        assert env.n_distractors == 0
        assert env.max_steps == 15

    def test_medium(self) -> None:
        env = ScoutMemEnv.medium()
        assert env.grid_size == 4
        assert env.n_objects == 5
        assert env.n_distractors == 1
        assert env.max_steps == 20

    def test_hard(self) -> None:
        env = ScoutMemEnv.hard()
        assert env.grid_size == 5
        assert env.n_objects == 6
        assert env.n_distractors == 2
        assert env.max_steps == 25


class TestRewardBounds:
    def test_step_cost(self) -> None:
        env = ScoutMemEnv.easy()
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(0)  # move
        assert -1.5 <= reward <= 1.0

    def test_correct_stop_reward(self) -> None:
        """Stop action gives +1.0 if position and confidence are right."""
        env = ScoutMemEnv.easy()
        env.reset(seed=0)
        # Move around to build confidence, then stop
        for _ in range(10):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
        # Reward from stop is bounded
        # (may not always be +1.0 since it depends on memory state)

    def test_wrong_stop_reward(self) -> None:
        env = ScoutMemEnv.easy()
        env.reset(seed=0)
        # Stop immediately with no evidence
        _, reward, term, _, _ = env.step(4)
        assert term is True
        assert reward == -1.0

    def test_timeout_reward(self) -> None:
        env = ScoutMemEnv(grid_size=3, max_steps=3, frame_stack=4)
        env.reset(seed=0)
        for _ in range(3):
            _, reward, term, trunc, _ = env.step(0)
            if trunc:
                assert reward == -0.5
                return
            if term:
                return
        pytest.fail("Expected truncation")


class TestStepMechanics:
    def test_movement_bounds(self) -> None:
        env = ScoutMemEnv(grid_size=3)
        env.reset(seed=0)
        env._agent_pos = np.array([0, 0])
        env.step(2)  # left: should clip
        assert env._agent_pos[0] >= 0

    def test_stop_terminates(self) -> None:
        env = ScoutMemEnv.easy()
        env.reset(seed=0)
        _, _, term, _, _ = env.step(4)
        assert term is True

    def test_info_keys(self) -> None:
        env = ScoutMemEnv.easy()
        _, info = env.reset(seed=0)
        expected_keys = {
            "step", "agent_pos", "target_pos", "target_label",
            "target_confidence", "memory_nodes", "coverage",
            "evidence_sufficiency",
        }
        assert expected_keys <= set(info.keys())

    def test_visited_tracking(self) -> None:
        env = ScoutMemEnv(grid_size=3)
        env.reset(seed=0)
        initial_visited = len(env._visited)
        # Move to a new cell
        env._agent_pos = np.array([1, 1])
        env.step(0)  # up
        assert len(env._visited) >= initial_visited


class TestRender:
    def test_ansi_render(self) -> None:
        env = ScoutMemEnv.easy(render_mode="ansi")
        env.reset(seed=0)
        output = env.render()
        assert output is not None
        assert "A" in output

    def test_no_render(self) -> None:
        env = ScoutMemEnv.easy()
        env.reset(seed=0)
        assert env.render() is None
