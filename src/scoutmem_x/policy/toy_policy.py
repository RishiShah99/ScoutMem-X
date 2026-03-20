from __future__ import annotations

from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.policy.actions import ActionType, AgentAction


def choose_toy_action(
    memory_snapshot: MemorySnapshot,
    max_steps: int,
    step_index: int,
) -> AgentAction:
    if memory_snapshot.evidence_sufficiency_score >= 0.8:
        target_id = memory_snapshot.nodes[-1].object_id if memory_snapshot.nodes else None
        return AgentAction(action_type=ActionType.STOP, target_id=target_id, cost=0.0)

    if step_index >= max_steps - 1:
        return AgentAction(
            action_type=ActionType.STOP,
            cost=0.0,
            metadata={"reason": "budget_exhausted"},
        )

    if memory_snapshot.nodes:
        return AgentAction(
            action_type=ActionType.INSPECT,
            target_id=memory_snapshot.nodes[-1].object_id,
            cost=0.2,
        )

    return AgentAction(action_type=ActionType.EXPLORE, cost=1.0)
