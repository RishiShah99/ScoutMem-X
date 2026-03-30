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
    decay_rate: float = 0.02,
) -> MemorySnapshot:
    """Build a new memory snapshot by merging detections with prior memory.

    Temporal decay: nodes not seen in the current step lose confidence
    proportional to how many steps have elapsed since last observation.
    This is what makes ScoutMem-X fundamentally different from a vector DB —
    old, unverified memories become less trustworthy over time.
    """
    current_step = observation.step_index
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
            existing_by_key[key] = _decay_unseen_node(node, current_step, decay_rate)

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


def _decay_unseen_node(
    node: MemoryNode, current_step: int, decay_rate: float
) -> MemoryNode:
    """Apply temporal decay to a node that wasn't observed this step.

    Confidence decays proportionally to how long ago the node was last seen.
    This models the real-world intuition: "I saw the keys on the counter
    20 minutes ago — but someone might have moved them since."

    A vector DB stores an embedding forever at full strength.
    ScoutMem-X's memory degrades — forcing the agent to re-verify
    old observations, which is what real perception requires.
    """
    steps_since_seen = max(0, current_step - node.last_seen_step)
    decay = decay_rate * steps_since_seen
    decayed_confidence = max(0.0, node.confidence - decay)

    if decayed_confidence >= 0.8:
        next_state = VisibilityState.VISIBLE
    elif decayed_confidence >= 0.5:
        next_state = VisibilityState.PREVIOUSLY_SEEN
    elif decayed_confidence >= 0.2:
        next_state = VisibilityState.UNCERTAIN
    else:
        next_state = VisibilityState.HYPOTHESIZED

    return MemoryNode(
        object_id=node.object_id,
        category=node.category,
        query_match_score=node.query_match_score,
        confidence=decayed_confidence,
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
