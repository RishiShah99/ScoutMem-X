from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionType(str, Enum):
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    MOVE_FORWARD = "move_forward"
    INSPECT = "inspect"
    REVISIT = "revisit"
    EXPLORE = "explore"
    STOP = "stop"


@dataclass(frozen=True)
class AgentAction:
    action_type: ActionType
    target_id: str | None = None
    cost: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost < 0.0:
            raise ValueError("cost must be non-negative")
