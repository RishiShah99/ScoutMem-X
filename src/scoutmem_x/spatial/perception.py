"""Noisy spatial perception — distance-based confidence with dropout and noise."""

from __future__ import annotations

import math
import random
from typing import Any

from scoutmem_x.perception.adapters import Detection
from scoutmem_x.spatial.world import SpatialWorld


class SpatialPerceptionAdapter:
    """Simulates noisy object detection based on distance from agent.

    This is what makes ScoutMem-X different from a vector database:
    - Objects far away are detected with LOW confidence
    - Objects are sometimes MISSED entirely (dropout)
    - Confidence has random noise (simulating real detector behavior)
    - The agent must observe from MULTIPLE positions to build certainty
    - A single observation is unreliable — memory aggregation is essential
    """

    def __init__(
        self,
        world: SpatialWorld,
        view_range: float = 8.0,
        dropout_rate: float = 0.15,
        noise_std: float = 0.08,
    ) -> None:
        self.world = world
        self.view_range = view_range
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std

    def predict(self, observation: Any, query: str) -> list[Detection]:
        agent_pos: tuple[float, float, float] = getattr(observation, "pose", (0, 0, 0))
        detections: list[Detection] = []

        for obj in self.world.objects:
            dist = _dist(agent_pos, obj.position)

            # Too far — can't see it
            if dist > self.view_range:
                continue

            # Random dropout — sometimes miss objects (real detectors do this)
            if random.random() < self.dropout_rate:
                continue

            # Distance-based confidence: close = high, far = low
            # Uses sqrt falloff so medium-distance objects are still detectable
            raw_conf = 0.95 * (1.0 - (dist / self.view_range) ** 0.6)

            # Add Gaussian noise (real detectors aren't deterministic)
            noisy_conf = raw_conf + random.gauss(0, self.noise_std)
            noisy_conf = max(0.05, min(0.95, noisy_conf))

            # Build a region label from proximity
            region = _infer_region(obj.position, self.world)

            detections.append(
                Detection(
                    label=obj.label,
                    score=round(noisy_conf, 3),
                    region=(0, 0, 64, 64),
                    metadata={
                        "query": query,
                        "source": "spatial_perception",
                        "region": region,
                        "target_label": query.replace("find the ", "").replace("find ", "").strip(),
                        "distance": f"{dist:.2f}",
                        "mesh_name": obj.mesh_name,
                    },
                )
            )

        return detections


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _infer_region(
    pos: tuple[float, float, float], world: SpatialWorld
) -> str:
    """Assign a rough region label based on position in the bounding box."""
    xmin, _, zmin = world.bounds_min
    xmax, _, zmax = world.bounds_max
    xfrac = (pos[0] - xmin) / max(xmax - xmin, 0.01)
    zfrac = (pos[2] - zmin) / max(zmax - zmin, 0.01)
    # Divide into quadrants
    col = "left" if xfrac < 0.5 else "right"
    row = "front" if zfrac < 0.5 else "back"
    return f"{row}_{col}"
