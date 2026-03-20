from __future__ import annotations

from dataclasses import dataclass, field

Pose3D = tuple[float, float, float]
ImageSize = tuple[int, int]


@dataclass(frozen=True)
class Observation:
    frame_id: str
    step_index: int
    pose: Pose3D
    heading_radians: float
    image_size: ImageSize
    rgb_path: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width, height = self.image_size
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if width <= 0 or height <= 0:
            raise ValueError("image_size values must be positive")
