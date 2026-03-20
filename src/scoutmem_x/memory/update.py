from __future__ import annotations

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.schema import MemoryNode, MemorySnapshot, VisibilityState
from scoutmem_x.perception.adapters import Detection


def build_memory_snapshot(
    observation: Observation,
    detections: list[Detection],
    target_label: str,
    previous_snapshot: MemorySnapshot | None = None,
) -> MemorySnapshot:
    previous_nodes = previous_snapshot.nodes if previous_snapshot is not None else ()
    previous_ids = {node.object_id for node in previous_nodes}

    new_nodes = tuple(
        _detection_to_memory_node(
            observation=observation,
            detection=detection,
            index=index,
            target_label=target_label,
        )
        for index, detection in enumerate(detections)
    )

    merged_nodes = previous_nodes + tuple(
        node for node in new_nodes if node.object_id not in previous_ids
    )
    best_score = max((node.confidence for node in merged_nodes), default=0.0)
    revisitable_ids = tuple(node.object_id for node in merged_nodes if node.confidence < 0.8)
    unexplored_regions = () if best_score >= 0.8 else ("forward_sector",)

    return MemorySnapshot(
        nodes=merged_nodes,
        unexplored_regions=unexplored_regions,
        revisitable_object_ids=revisitable_ids,
        evidence_sufficiency_score=min(best_score, 1.0),
    )


def _detection_to_memory_node(
    observation: Observation,
    detection: Detection,
    index: int,
    target_label: str,
) -> MemoryNode:
    region_name = detection.metadata.get("region", "unknown_region")
    return MemoryNode(
        object_id=f"{observation.frame_id}-{index}",
        category=detection.label,
        query_match_score=detection.score,
        confidence=detection.score,
        last_seen_step=observation.step_index,
        visibility_state=_visibility_state_for_score(detection.score),
        position_estimate=observation.pose,
        room_or_region_estimate=region_name,
        supporting_frames=(observation.frame_id,),
        visual_embedding=detection.embedding,
        metadata={
            "query": detection.metadata.get("query", ""),
            "target_label": detection.metadata.get("target_label", target_label),
        },
    )


def _visibility_state_for_score(score: float) -> VisibilityState:
    if score >= 0.8:
        return VisibilityState.VISIBLE
    if score >= 0.5:
        return VisibilityState.UNCERTAIN
    return VisibilityState.HYPOTHESIZED
