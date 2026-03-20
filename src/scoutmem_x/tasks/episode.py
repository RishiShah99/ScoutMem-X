from __future__ import annotations

from dataclasses import dataclass, field

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.perception.adapters import Detection
from scoutmem_x.policy.actions import AgentAction


@dataclass(frozen=True)
class EpisodeStepRecord:
    observation: Observation
    detections: tuple[Detection, ...]
    action: AgentAction
    memory_snapshot: MemorySnapshot
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodeTrace:
    episode_id: str
    query: str
    steps: tuple[EpisodeStepRecord, ...] = ()
    success: bool | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def final_action(self) -> AgentAction | None:
        if not self.steps:
            return None
        return self.steps[-1].action
