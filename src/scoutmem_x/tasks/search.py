from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.config import AppConfig
from scoutmem_x.env import GridSearchEnv, SearchSceneSpec
from scoutmem_x.memory import MemorySnapshot, build_memory_snapshot, retrieve_best_node
from scoutmem_x.perception import Detection, OraclePerceptionAdapter, PerceptionAdapter
from scoutmem_x.policy import (
    ActionType,
    AgentAction,
    choose_active_evidence_action,
    choose_passive_memory_action,
    choose_reactive_action,
    estimate_uncertainty,
)
from scoutmem_x.tasks.episode import EpisodeStepRecord, EpisodeTrace


@dataclass(frozen=True)
class SearchEpisodeResult:
    trace: EpisodeTrace
    success: bool
    steps_taken: int
    scene_id: str
    split: str
    final_memory: MemorySnapshot


def run_passive_memory_search_episode(
    scene: SearchSceneSpec,
    config: AppConfig,
    perception_adapter: PerceptionAdapter | None = None,
) -> SearchEpisodeResult:
    return _run_search_episode(
        scene=scene,
        config=config,
        perception_adapter=perception_adapter,
        strategy="passive_memory",
    )


def run_active_evidence_search_episode(
    scene: SearchSceneSpec,
    config: AppConfig,
    perception_adapter: PerceptionAdapter | None = None,
) -> SearchEpisodeResult:
    return _run_search_episode(
        scene=scene,
        config=config,
        perception_adapter=perception_adapter,
        strategy="active_evidence",
    )


def run_reactive_search_episode(
    scene: SearchSceneSpec,
    config: AppConfig,
    perception_adapter: PerceptionAdapter | None = None,
) -> SearchEpisodeResult:
    return _run_search_episode(
        scene=scene,
        config=config,
        perception_adapter=perception_adapter,
        strategy="reactive",
    )


def _run_search_episode(
    scene: SearchSceneSpec,
    config: AppConfig,
    perception_adapter: PerceptionAdapter | None,
    strategy: str,
) -> SearchEpisodeResult:
    env = GridSearchEnv(scene=scene)
    adapter = perception_adapter or OraclePerceptionAdapter()
    observation = env.reset()
    step_records: list[EpisodeStepRecord] = []
    found_target = False
    memory_snapshot = MemorySnapshot()

    for step_index in range(config.max_steps):
        detections = adapter.predict(observation=observation, query=config.query)
        memory_snapshot = build_memory_snapshot(
            observation=observation,
            detections=detections,
            target_label=config.target_label,
            previous_snapshot=memory_snapshot if strategy != "reactive" else None,
        )
        if strategy == "passive_memory":
            action = choose_passive_memory_action(
                memory_snapshot=memory_snapshot,
                target_label=config.target_label,
                stop_threshold=config.stop_threshold,
                max_steps=config.max_steps,
                step_index=step_index,
            )
        elif strategy == "active_evidence":
            action = choose_active_evidence_action(
                memory_snapshot=memory_snapshot,
                detections=detections,
                target_label=config.target_label,
                stop_threshold=config.stop_threshold,
                max_steps=config.max_steps,
                step_index=step_index,
            )
        else:
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
                notes=_build_step_notes(
                    scene=scene,
                    strategy=strategy,
                    memory_snapshot=memory_snapshot,
                    detections=detections,
                    target_label=config.target_label,
                    stop_threshold=config.stop_threshold,
                ),
            )
        )
        transition = env.step(action=action, step_index=step_index)
        observation = transition.observation
        found_target = _is_episode_successful(
            scene=scene,
            action=action,
            memory_snapshot=memory_snapshot,
            transition_found_target=transition.found_target,
        )
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
        final_memory=memory_snapshot,
    )


def _is_episode_successful(
    scene: SearchSceneSpec,
    action: AgentAction,
    memory_snapshot: MemorySnapshot,
    transition_found_target: bool,
) -> bool:
    if transition_found_target:
        return True
    if action.action_type != ActionType.STOP:
        return False
    best_node = retrieve_best_node(memory_snapshot, scene.target_label)
    if best_node is None:
        return False
    return best_node.confidence >= memory_snapshot.evidence_sufficiency_score >= 0.8


def _build_step_notes(
    scene: SearchSceneSpec,
    strategy: str,
    memory_snapshot: MemorySnapshot,
    detections: list[Detection],
    target_label: str,
    stop_threshold: float,
) -> tuple[str, ...]:
    uncertainty = estimate_uncertainty(
        memory_snapshot=memory_snapshot,
        detections=detections,
        target_label=target_label,
        stop_threshold=stop_threshold,
    )
    return (
        scene.scene_id,
        scene.split,
        f"strategy:{strategy}",
        f"confidence:{uncertainty.confidence:.2f}",
        f"evidence_count:{uncertainty.evidence_count}",
    )
