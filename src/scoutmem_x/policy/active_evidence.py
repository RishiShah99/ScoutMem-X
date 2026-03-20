from __future__ import annotations

from scoutmem_x.memory import MemorySnapshot
from scoutmem_x.perception import Detection
from scoutmem_x.policy.actions import ActionType, AgentAction
from scoutmem_x.policy.uncertainty import estimate_uncertainty


def choose_active_evidence_action(
    memory_snapshot: MemorySnapshot,
    detections: list[Detection],
    target_label: str,
    stop_threshold: float,
    max_steps: int,
    step_index: int,
) -> AgentAction:
    uncertainty = estimate_uncertainty(
        memory_snapshot=memory_snapshot,
        detections=detections,
        target_label=target_label,
        stop_threshold=stop_threshold,
    )

    if uncertainty.stop_recommended:
        return AgentAction(
            action_type=ActionType.STOP,
            target_id=uncertainty.target_object_id,
            cost=0.0,
            metadata={"reason": "uncertainty_satisfied"},
        )

    if uncertainty.inspect_recommended:
        return AgentAction(
            action_type=ActionType.INSPECT,
            target_id=uncertainty.target_object_id,
            cost=0.1,
            metadata={"reason": "gather_more_evidence"},
        )

    if step_index >= max_steps - 1:
        return AgentAction(
            action_type=ActionType.STOP,
            target_id=uncertainty.target_object_id,
            cost=0.0,
            metadata={"reason": "budget_exhausted"},
        )

    if uncertainty.revisit_candidates and not uncertainty.target_visible_now:
        return AgentAction(
            action_type=ActionType.REVISIT,
            target_id=uncertainty.revisit_candidates[0],
            cost=0.5,
            metadata={"reason": "revisit_weak_evidence"},
        )

    return AgentAction(action_type=ActionType.MOVE_FORWARD, cost=1.0)
