from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    region: tuple[int, int, int, int] | None = None
    embedding: tuple[float, ...] = ()
    mask: tuple[tuple[int, int], ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PerceptionAdapter(Protocol):
    def predict(self, observation: Any, query: str) -> list[Detection]:
        """Return normalized detections for an observation and query."""
        ...


class MockPerceptionAdapter:
    def predict(self, observation: Any, query: str) -> list[Detection]:
        del observation
        return [
            Detection(
                label="target_hint",
                score=0.5,
                region=(0, 0, 10, 10),
                metadata={"query": query, "source": "mock"},
            )
        ]
