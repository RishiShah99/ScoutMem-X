from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.config import AppConfig
from scoutmem_x.env import Observation
from scoutmem_x.memory import MemorySnapshot, build_memory_snapshot
from scoutmem_x.perception.adapters import MockPerceptionAdapter, PerceptionAdapter
from scoutmem_x.policy import ActionType, choose_toy_action
from scoutmem_x.tasks.episode import EpisodeStepRecord, EpisodeTrace


@dataclass(frozen=True)
class ToyEpisodeResult:
    trace: EpisodeTrace
    final_memory: MemorySnapshot


def run_toy_episode(
    config: AppConfig,
    perception_adapter: PerceptionAdapter | None = None,
) -> ToyEpisodeResult:
    adapter = perception_adapter or MockPerceptionAdapter()
    memory_snapshot = MemorySnapshot()
    step_records: list[EpisodeStepRecord] = []

    for step_index in range(config.max_steps):
        observation = _build_observation(step_index=step_index)
        detections = adapter.predict(observation=observation, query=config.query)
        memory_snapshot = build_memory_snapshot(
            observation=observation,
            detections=detections,
            target_label=config.target_label,
            previous_snapshot=memory_snapshot,
        )
        action = choose_toy_action(
            memory_snapshot=memory_snapshot,
            max_steps=config.max_steps,
            step_index=step_index,
        )
        step_records.append(
            EpisodeStepRecord(
                observation=observation,
                detections=tuple(detections),
                action=action,
                memory_snapshot=memory_snapshot,
                notes=(f"toy_step_{step_index}",),
            )
        )
        if action.action_type == ActionType.STOP:
            break

    trace = EpisodeTrace(
        episode_id=f"{config.phase}-{config.subphase}",
        query=config.query,
        steps=tuple(step_records),
        success=memory_snapshot.evidence_sufficiency_score >= config.stop_threshold,
        metadata={"mode": config.mode, "target_label": config.target_label},
    )
    return ToyEpisodeResult(trace=trace, final_memory=memory_snapshot)


def _build_observation(step_index: int) -> Observation:
    return Observation(
        frame_id=f"frame-{step_index}",
        step_index=step_index,
        pose=(float(step_index), 0.0, 0.0),
        heading_radians=0.25 * step_index,
        image_size=(128, 128),
        rgb_path=f"outputs/frame-{step_index}.png",
        metadata={"scene_id": "toy_room"},
    )
