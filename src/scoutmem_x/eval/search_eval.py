from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.config import AppConfig
from scoutmem_x.env import SearchSceneSpec, load_default_scenes
from scoutmem_x.tasks import run_reactive_search_episode


@dataclass(frozen=True)
class EvalEpisodeBrief:
    scene_id: str
    split: str
    success: bool
    steps_taken: int


@dataclass(frozen=True)
class EvalSummary:
    phase: str
    split: str
    total_episodes: int
    success_rate: float
    average_steps: float
    episode_briefs: tuple[EvalEpisodeBrief, ...]


def evaluate_reactive_baseline(config: AppConfig) -> EvalSummary:
    scenes = _select_scenes(config)
    results = [run_reactive_search_episode(scene=scene, config=config) for scene in scenes]
    episode_briefs = tuple(
        EvalEpisodeBrief(
            scene_id=result.scene_id,
            split=result.split,
            success=result.success,
            steps_taken=result.steps_taken,
        )
        for result in results
    )
    total = len(results)
    success_count = sum(1 for result in results if result.success)
    total_steps = sum(result.steps_taken for result in results)
    return EvalSummary(
        phase=config.phase,
        split=config.split,
        total_episodes=total,
        success_rate=(success_count / total) if total else 0.0,
        average_steps=(total_steps / total) if total else 0.0,
        episode_briefs=episode_briefs,
    )


def _select_scenes(config: AppConfig) -> tuple[SearchSceneSpec, ...]:
    scenes = load_default_scenes()
    filtered = tuple(scene for scene in scenes if scene.split == config.split)
    if config.scene_ids:
        requested_ids = set(config.scene_ids)
        filtered = tuple(scene for scene in filtered if scene.scene_id in requested_ids)
    return filtered
