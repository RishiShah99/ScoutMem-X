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
        step_index = getattr(observation, "step_index", 0)
        score = min(0.35 + 0.25 * step_index, 0.95)
        return [
            Detection(
                label="red mug",
                score=score,
                region=(0, 0, 10, 10),
                metadata={
                    "query": query,
                    "source": "mock",
                    "region": "countertop",
                    "target_label": "red mug",
                },
            )
        ]


class OraclePerceptionAdapter:
    def predict(self, observation: Any, query: str) -> list[Detection]:
        metadata = getattr(observation, "metadata", {})
        label = metadata.get("visible_label", "")
        score = float(metadata.get("visible_score", "0.0"))
        if not label or score <= 0.0:
            return []

        return [
            Detection(
                label=label,
                score=score,
                region=(0, 0, 12, 12),
                metadata={
                    "query": query,
                    "source": "oracle",
                    "region": metadata.get("visible_region", "forward_cell"),
                    "target_label": query.replace("find the ", ""),
                },
            )
        ]
