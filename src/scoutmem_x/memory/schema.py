from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

Position3D = tuple[float, float, float]
RegionHint = tuple[str, ...]


class VisibilityState(str, Enum):
    VISIBLE = "visible"
    PREVIOUSLY_SEEN = "previously_seen"
    UNCERTAIN = "uncertain"
    HYPOTHESIZED = "hypothesized"


@dataclass(frozen=True)
class Relation:
    relation_type: str
    target: str


@dataclass(frozen=True)
class MemoryNode:
    object_id: str
    category: str
    query_match_score: float
    confidence: float
    last_seen_step: int
    visibility_state: VisibilityState
    position_estimate: Position3D | None = None
    room_or_region_estimate: str | None = None
    supporting_frames: tuple[str, ...] = ()
    supporting_crops: tuple[str, ...] = ()
    visual_embedding: tuple[float, ...] = ()
    relations: tuple[Relation, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.query_match_score <= 1.0:
            raise ValueError("query_match_score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.last_seen_step < 0:
            raise ValueError("last_seen_step must be non-negative")


@dataclass(frozen=True)
class MemorySnapshot:
    nodes: tuple[MemoryNode, ...] = ()
    unexplored_regions: RegionHint = ()
    revisitable_object_ids: tuple[str, ...] = ()
    evidence_sufficiency_score: float = 0.0
    target_object_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.evidence_sufficiency_score <= 1.0:
            raise ValueError("evidence_sufficiency_score must be between 0.0 and 1.0")
