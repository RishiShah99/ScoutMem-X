from __future__ import annotations

from scoutmem_x.memory import MemorySnapshot, retrieve_best_node
from scoutmem_x.policy.actions import ActionType, AgentAction


def choose_passive_memory_action(
    memory_snapshot: MemorySnapshot,
    target_label: str,
    stop_threshold: float,
    max_steps: int,
    step_index: int,
) -> AgentAction:
    best_node = retrieve_best_node(memory_snapshot, target_label)
    if best_node is not None and best_node.confidence >= stop_threshold:
        return AgentAction(
            action_type=ActionType.STOP,
            target_id=best_node.object_id,
            cost=0.0,
            metadata={"reason": "memory_confident"},
        )

    if step_index >= max_steps - 1:
        return AgentAction(
            action_type=ActionType.STOP,
            target_id=best_node.object_id if best_node is not None else None,
            cost=0.0,
            metadata={"reason": "budget_exhausted"},
        )

    return AgentAction(action_type=ActionType.MOVE_FORWARD, cost=1.0)
