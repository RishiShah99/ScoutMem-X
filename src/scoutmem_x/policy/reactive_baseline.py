from __future__ import annotations

from scoutmem_x.perception.adapters import Detection
from scoutmem_x.policy.actions import ActionType, AgentAction


def choose_reactive_action(
    detections: list[Detection],
    target_label: str,
    stop_threshold: float,
    max_steps: int,
    step_index: int,
) -> AgentAction:
    best_target = max(
        (detection for detection in detections if detection.label == target_label),
        key=lambda detection: detection.score,
        default=None,
    )
    if best_target is not None and best_target.score >= stop_threshold:
        return AgentAction(
            action_type=ActionType.STOP,
            cost=0.0,
            metadata={"reason": "target_visible"},
        )

    if step_index >= max_steps - 1:
        return AgentAction(
            action_type=ActionType.STOP,
            cost=0.0,
            metadata={"reason": "budget_exhausted"},
        )

    return AgentAction(action_type=ActionType.MOVE_FORWARD, cost=1.0)
