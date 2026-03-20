from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.memory import MemorySnapshot, retrieve_best_node
from scoutmem_x.perception import Detection


@dataclass(frozen=True)
class UncertaintyEstimate:
    target_object_id: str | None
    confidence: float
    evidence_count: int
    target_visible_now: bool
    inspect_recommended: bool
    stop_recommended: bool
    revisit_candidates: tuple[str, ...]


def estimate_uncertainty(
    memory_snapshot: MemorySnapshot,
    detections: list[Detection],
    target_label: str,
    stop_threshold: float,
) -> UncertaintyEstimate:
    best_node = retrieve_best_node(memory_snapshot, target_label)
    current_target_detection = next(
        (detection for detection in detections if detection.label == target_label),
        None,
    )
    confidence = best_node.confidence if best_node is not None else 0.0
    evidence_count = len(best_node.supporting_frames) if best_node is not None else 0
    target_visible_now = (
        current_target_detection is not None and current_target_detection.score > 0.0
    )
    visible_confidence = (
        current_target_detection.score if current_target_detection is not None else 0.0
    )
    reference_confidence = max(confidence, visible_confidence)
    stop_recommended = confidence >= stop_threshold
    inspect_recommended = (
        target_visible_now
        and not stop_recommended
        and reference_confidence >= 0.45
        and evidence_count < 3
    )

    return UncertaintyEstimate(
        target_object_id=(
            best_node.object_id
            if best_node is not None
            else memory_snapshot.target_object_id
        ),
        confidence=confidence,
        evidence_count=evidence_count,
        target_visible_now=target_visible_now,
        inspect_recommended=inspect_recommended,
        stop_recommended=stop_recommended,
        revisit_candidates=memory_snapshot.revisitable_object_ids,
    )
