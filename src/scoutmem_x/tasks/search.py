from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.config import AppConfig
from scoutmem_x.env import GridSearchEnv, SearchSceneSpec
from scoutmem_x.memory import build_memory_snapshot
from scoutmem_x.perception import OraclePerceptionAdapter, PerceptionAdapter
from scoutmem_x.policy import choose_reactive_action
from scoutmem_x.tasks.episode import EpisodeStepRecord, EpisodeTrace


@dataclass(frozen=True)
class SearchEpisodeResult:
    trace: EpisodeTrace
    success: bool
    steps_taken: int
    scene_id: str
    split: str


def run_reactive_search_episode(
    scene: SearchSceneSpec,
    config: AppConfig,
    perception_adapter: PerceptionAdapter | None = None,
) -> SearchEpisodeResult:
    env = GridSearchEnv(scene=scene)
    adapter = perception_adapter or OraclePerceptionAdapter()
    observation = env.reset()
    step_records: list[EpisodeStepRecord] = []
    found_target = False

    for step_index in range(config.max_steps):
        detections = adapter.predict(observation=observation, query=config.query)
        memory_snapshot = build_memory_snapshot(
            observation=observation,
            detections=detections,
            target_label=config.target_label,
            previous_snapshot=None,
        )
        action = choose_reactive_action(
            detections=detections,
            target_label=config.target_label,
            stop_threshold=config.stop_threshold,
            max_steps=config.max_steps,
            step_index=step_index,
        )
        step_records.append(
            EpisodeStepRecord(
                observation=observation,
                detections=tuple(detections),
                action=action,
                memory_snapshot=memory_snapshot,
                notes=(scene.scene_id, scene.split),
            )
        )
        transition = env.step(action=action, step_index=step_index)
        observation = transition.observation
        found_target = transition.found_target
        if transition.done:
            break

    trace = EpisodeTrace(
        episode_id=f"{scene.scene_id}-{config.phase}",
        query=config.query,
        steps=tuple(step_records),
        success=found_target,
        metadata={"scene_id": scene.scene_id, "split": scene.split, "mode": config.mode},
    )
    return SearchEpisodeResult(
        trace=trace,
        success=found_target,
        steps_taken=trace.step_count,
        scene_id=scene.scene_id,
        split=scene.split,
    )
