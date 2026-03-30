"""Curriculum callback for Stable-Baselines3 PPO.

Automatically advances environment difficulty as the agent's eval
success rate crosses preset thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from scoutmem_x.rl.env import ScoutMemEnv


@dataclass(frozen=True)
class Level:
    name: str
    factory: str  # ScoutMemEnv classmethod name
    promotion_threshold: float | None  # None = final level


LEVELS: list[Level] = [
    Level(name="easy", factory="easy", promotion_threshold=0.55),
    Level(name="medium", factory="medium", promotion_threshold=0.50),
    Level(name="hard", factory="hard", promotion_threshold=None),
]


@dataclass
class LevelTransition:
    timestep: int
    from_level: str
    to_level: str
    success_rate: float


class CurriculumCallback(BaseCallback):
    """Advances training difficulty when eval success rate exceeds a threshold.

    Runs ``n_eval_episodes`` rollouts on a private eval env every
    ``check_freq`` timesteps. When the success rate exceeds the current
    level's promotion threshold, the training ``VecEnv`` is rebuilt at
    the next difficulty.
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        n_eval_episodes: int = 30,
        n_envs: int = 8,
        seed: int = 42,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_envs = n_envs
        self.seed = seed

        self._level_idx: int = 0
        self._transitions: list[LevelTransition] = []
        self._last_check: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_current_level(self) -> str:
        return LEVELS[self._level_idx].name

    @property
    def transitions(self) -> list[LevelTransition]:
        return list(self._transitions)

    # ------------------------------------------------------------------
    # BaseCallback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self.check_freq:
            return True
        self._last_check = self.num_timesteps

        level = LEVELS[self._level_idx]
        if level.promotion_threshold is None:
            return True  # already at hardest level

        success_rate = self._evaluate()
        if self.verbose:
            print(
                f"[Curriculum] t={self.num_timesteps:,}  "
                f"level={level.name}  success={success_rate:.1%}"
            )

        if success_rate >= level.promotion_threshold:
            self._advance(success_rate)

        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate(self) -> float:
        level = LEVELS[self._level_idx]
        factory = getattr(ScoutMemEnv, level.factory)

        # Wrap in VecNormalize to match training observations
        eval_venv = make_vec_env(lambda: factory(seed=self.seed + 9999), n_envs=1)
        eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, training=False)
        # Copy running stats from training env
        train_venv = self.model.get_env()  # type: ignore[union-attr]
        if isinstance(train_venv, VecNormalize):
            eval_venv.obs_rms = train_venv.obs_rms
            eval_venv.ret_rms = train_venv.ret_rms

        successes = 0
        for _ in range(self.n_eval_episodes):
            obs = eval_venv.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[union-attr]
                obs, rewards, dones, infos = eval_venv.step(action)
                done = bool(dones[0])
            # Correct stop gives reward > 0; wrong stop or timeout gives reward <= 0
            if float(rewards[0]) > 0:  # type: ignore[possibly-undefined]
                successes += 1

        eval_venv.close()
        return successes / self.n_eval_episodes

    def _advance(self, success_rate: float) -> None:
        old_level = LEVELS[self._level_idx]
        self._level_idx += 1
        new_level = LEVELS[self._level_idx]

        self._transitions.append(
            LevelTransition(
                timestep=self.num_timesteps,
                from_level=old_level.name,
                to_level=new_level.name,
                success_rate=success_rate,
            )
        )

        print(
            f"[Curriculum] ADVANCING {old_level.name} -> {new_level.name}  "
            f"(success={success_rate:.1%} >= {old_level.promotion_threshold:.0%})  "
            f"t={self.num_timesteps:,}"
        )

        new_factory = getattr(ScoutMemEnv, new_level.factory)
        new_train_env = make_vec_env(
            lambda: new_factory(seed=self.seed),
            n_envs=self.n_envs,
        )
        new_train_env = VecNormalize(
            new_train_env, norm_obs=True, norm_reward=False, clip_reward=5.0,
        )

        self.model.set_env(new_train_env)  # type: ignore[union-attr]
