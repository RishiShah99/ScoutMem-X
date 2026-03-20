from __future__ import annotations

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.memory.schema import MemoryNode, MemorySnapshot, VisibilityState
from scoutmem_x.perception.adapters import Detection


def build_memory_snapshot(
    observation: Observation,
    detections: list[Detection],
    target_label: str,
    previous_snapshot: MemorySnapshot | None = None,
) -> MemorySnapshot:
    existing_by_key = {
        _memory_key(node.category, node.room_or_region_estimate): node
        for node in (previous_snapshot.nodes if previous_snapshot is not None else ())
    }
    seen_keys: set[tuple[str, str | None]] = set()

    for index, detection in enumerate(detections):
        key = _memory_key(detection.label, detection.metadata.get("region"))
        existing = existing_by_key.get(key)
        existing_by_key[key] = _merge_detection_into_node(
            observation=observation,
            detection=detection,
            index=index,
            target_label=target_label,
            previous_node=existing,
        )
        seen_keys.add(key)

    for key, node in tuple(existing_by_key.items()):
        if key not in seen_keys:
            existing_by_key[key] = _mark_node_previously_seen(node)

    merged_nodes = tuple(
        sorted(existing_by_key.values(), key=lambda node: (node.last_seen_step, node.object_id))
    )
    best_node = retrieve_best_node(MemorySnapshot(nodes=merged_nodes), target_label)
    best_score = best_node.confidence if best_node is not None else 0.0
    revisitable_ids = tuple(
        node.object_id
        for node in merged_nodes
        if node.category == target_label and node.confidence < 0.8
    )
    unexplored_regions = () if best_score >= 0.8 else ("forward_sector",)

    return MemorySnapshot(
        nodes=merged_nodes,
        unexplored_regions=unexplored_regions,
        revisitable_object_ids=revisitable_ids,
        evidence_sufficiency_score=min(best_score, 1.0),
        target_object_id=best_node.object_id if best_node is not None else None,
    )


def _merge_detection_into_node(
    observation: Observation,
    detection: Detection,
    index: int,
    target_label: str,
    previous_node: MemoryNode | None,
) -> MemoryNode:
    region_name = detection.metadata.get("region", "unknown_region")
    object_id = (
        previous_node.object_id
        if previous_node is not None
        else f"{detection.label}-{region_name}"
    )
    prior_frames = previous_node.supporting_frames if previous_node is not None else ()
    prior_confidence = previous_node.confidence if previous_node is not None else 0.0
    aggregated_confidence = 1.0 - ((1.0 - prior_confidence) * (1.0 - detection.score))
    supporting_frames = _append_unique(prior_frames, observation.frame_id)
    query_score = (
        max(previous_node.query_match_score, detection.score)
        if previous_node
        else detection.score
    )
    return MemoryNode(
        object_id=object_id,
        category=detection.label,
        query_match_score=query_score,
        confidence=min(aggregated_confidence, 1.0),
        last_seen_step=observation.step_index,
        visibility_state=_visibility_state_for_score(aggregated_confidence),
        position_estimate=observation.pose,
        room_or_region_estimate=region_name,
        supporting_frames=supporting_frames,
        supporting_crops=previous_node.supporting_crops if previous_node is not None else (),
        visual_embedding=detection.embedding,
        relations=previous_node.relations if previous_node is not None else (),
        metadata={
            "query": detection.metadata.get("query", ""),
            "target_label": detection.metadata.get("target_label", target_label),
        },
    )


def _mark_node_previously_seen(node: MemoryNode) -> MemoryNode:
    if node.visibility_state == VisibilityState.VISIBLE:
        next_state = VisibilityState.PREVIOUSLY_SEEN
    else:
        next_state = node.visibility_state
    return MemoryNode(
        object_id=node.object_id,
        category=node.category,
        query_match_score=node.query_match_score,
        confidence=node.confidence,
        last_seen_step=node.last_seen_step,
        visibility_state=next_state,
        position_estimate=node.position_estimate,
        room_or_region_estimate=node.room_or_region_estimate,
        supporting_frames=node.supporting_frames,
        supporting_crops=node.supporting_crops,
        visual_embedding=node.visual_embedding,
        relations=node.relations,
        metadata=node.metadata,
    )


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    if value in values:
        return values
    return values + (value,)


def _memory_key(category: str, region_name: str | None) -> tuple[str, str | None]:
    return (category, region_name)


def _visibility_state_for_score(score: float) -> VisibilityState:
    if score >= 0.8:
        return VisibilityState.VISIBLE
    if score >= 0.5:
        return VisibilityState.UNCERTAIN
    return VisibilityState.HYPOTHESIZED
