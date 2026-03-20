from __future__ import annotations

from dataclasses import dataclass, field

from scoutmem_x.env.observation import Observation


@dataclass(frozen=True)
class SearchSceneSpec:
    scene_id: str
    split: str
    length: int
    target_label: str
    target_position: int
    target_visibility: dict[int, float] = field(default_factory=dict)
    distractors: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.length <= 1:
            raise ValueError("length must be greater than 1")
        if not 0 <= self.target_position < self.length:
            raise ValueError("target_position must fall within the scene length")
        for position, score in self.target_visibility.items():
            if not 0 <= position < self.length:
                raise ValueError("target_visibility positions must fall within the scene length")
            if not 0.0 <= score <= 1.0:
                raise ValueError("target_visibility scores must be between 0.0 and 1.0")


@dataclass(frozen=True)
class EnvironmentStep:
    observation: Observation
    reward: float
    done: bool
    found_target: bool
