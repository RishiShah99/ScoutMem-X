"""2D apartment grid world for the ScoutMem-X interactive demo."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from scoutmem_x.env.observation import Observation
from scoutmem_x.perception.adapters import Detection


@dataclass(frozen=True)
class RoomSpec:
    """A room in the 2D apartment."""

    name: str
    room_type: str
    objects: dict[str, float] = field(default_factory=dict)
    false_positives: dict[str, tuple[str, float]] = field(default_factory=dict)
    color: str = "#2d2d3f"


@dataclass(frozen=True)
class ApartmentSpec:
    """Layout of a 2D apartment grid."""

    apartment_id: str
    width: int
    height: int
    rooms: dict[tuple[int, int], RoomSpec] = field(default_factory=dict)
    walls: frozenset[tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=frozenset
    )


class GridWorld2D:
    """2D grid world where each cell is a room containing objects."""

    def __init__(self, spec: ApartmentSpec) -> None:
        self.spec = spec
        self.agent_pos: tuple[int, int] = (0, 0)
        self.step_count: int = 0
        self.visited: set[tuple[int, int]] = set()
        self._inspecting: bool = False
        self._visit_counts: dict[tuple[int, int], int] = {}

    def reset(self, start: tuple[int, int] = (0, 0)) -> Observation:
        self.agent_pos = start
        self.step_count = 0
        self.visited = {start}
        self._inspecting = False
        self._visit_counts = {start: 1}
        return self._build_observation()

    def move_to(self, pos: tuple[int, int]) -> Observation:
        """Move agent to an adjacent room."""
        if pos not in self._neighbors_of(self.agent_pos):
            raise ValueError(f"Cannot move from {self.agent_pos} to {pos}")
        self.agent_pos = pos
        self.step_count += 1
        self.visited.add(pos)
        self._visit_counts[pos] = self._visit_counts.get(pos, 0) + 1
        self._inspecting = False
        return self._build_observation()

    def inspect(self) -> Observation:
        """Stay in current room and observe more carefully."""
        self.step_count += 1
        self._inspecting = True
        return self._build_observation()

    def get_neighbors(self) -> list[tuple[int, int]]:
        return self._neighbors_of(self.agent_pos)

    def get_nearest_unvisited(self) -> tuple[int, int] | None:
        """BFS to find the first step toward the nearest unvisited room."""
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque(
            [(self.agent_pos, [self.agent_pos])]
        )
        seen = {self.agent_pos}
        while queue:
            pos, path = queue.popleft()
            if pos not in self.visited and pos != self.agent_pos:
                return path[1] if len(path) > 1 else pos
            for neighbor in self._neighbors_of(pos):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def find_path(self, target: tuple[int, int]) -> list[tuple[int, int]]:
        """BFS shortest path from agent to target."""
        if self.agent_pos == target:
            return [self.agent_pos]
        queue: deque[tuple[tuple[int, int], list[tuple[int, int]]]] = deque(
            [(self.agent_pos, [self.agent_pos])]
        )
        seen = {self.agent_pos}
        while queue:
            pos, path = queue.popleft()
            for neighbor in self._neighbors_of(pos):
                if neighbor not in seen:
                    new_path = path + [neighbor]
                    if neighbor == target:
                        return new_path
                    seen.add(neighbor)
                    queue.append((neighbor, new_path))
        return []

    @property
    def current_room(self) -> RoomSpec | None:
        return self.spec.rooms.get(self.agent_pos)

    @property
    def all_explored(self) -> bool:
        return self.visited >= set(self.spec.rooms.keys())

    def _neighbors_of(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.spec.rooms:
                pair = (pos, (nx, ny))
                rev_pair = ((nx, ny), pos)
                if pair not in self.spec.walls and rev_pair not in self.spec.walls:
                    neighbors.append((nx, ny))
        return neighbors

    def _build_observation(self) -> Observation:
        room = self.current_room
        room_name = room.name if room else "unknown"
        room_type = room.room_type if room else "unknown"

        visible_objects: dict[str, str] = {}
        if room:
            visit_count = self._visit_counts.get(self.agent_pos, 1)
            for label, base_score in room.objects.items():
                if self._inspecting:
                    score = base_score
                elif visit_count <= 1:
                    score = base_score * 0.65
                else:
                    score = base_score * 0.75
                visible_objects[f"obj_{len(visible_objects)}"] = (
                    f"{label}|{min(score, 1.0):.3f}"
                )
            for _real_label, (false_label, false_score) in room.false_positives.items():
                if self._inspecting:
                    score = false_score
                elif visit_count <= 1:
                    score = false_score * 0.65
                else:
                    score = false_score * 0.75
                visible_objects[f"obj_{len(visible_objects)}"] = (
                    f"{false_label}|{min(score, 1.0):.3f}"
                )

        metadata: dict[str, str] = {
            "scene_id": self.spec.apartment_id,
            "room_name": room_name,
            "room_type": room_type,
            "agent_x": str(self.agent_pos[0]),
            "agent_y": str(self.agent_pos[1]),
            "inspecting": str(self._inspecting),
            "visit_count": str(self._visit_counts.get(self.agent_pos, 0)),
            "obj_count": str(len(visible_objects)),
        }
        for key, val in visible_objects.items():
            metadata[key] = val

        return Observation(
            frame_id=f"{self.spec.apartment_id}-step-{self.step_count}",
            step_index=self.step_count,
            pose=(float(self.agent_pos[0]), float(self.agent_pos[1]), 0.0),
            heading_radians=0.0,
            image_size=(256, 256),
            metadata=metadata,
        )


class Oracle2DAdapter:
    """Reads detections from 2D observation metadata."""

    def predict(self, observation: Any, query: str) -> list[Detection]:
        metadata = getattr(observation, "metadata", {})
        obj_count = int(metadata.get("obj_count", "0"))
        room_name = metadata.get("room_name", "unknown")
        detections: list[Detection] = []
        for i in range(obj_count):
            raw = metadata.get(f"obj_{i}", "")
            if "|" not in raw:
                continue
            label, score_str = raw.rsplit("|", 1)
            score = float(score_str)
            if score <= 0.0:
                continue
            detections.append(
                Detection(
                    label=label,
                    score=score,
                    region=(0, 0, 64, 64),
                    metadata={
                        "query": query,
                        "source": "oracle_2d",
                        "region": room_name,
                        "target_label": query.replace("find the ", ""),
                    },
                )
            )
        return detections


# ---------------------------------------------------------------------------
# Default apartment layout
# ---------------------------------------------------------------------------

DEMO_APARTMENT = ApartmentSpec(
    apartment_id="modern_apartment",
    width=4,
    height=3,
    rooms={
        (0, 0): RoomSpec(
            "Kitchen", "kitchen",
            objects={"coffee maker": 0.90, "mug": 0.85, "cutting board": 0.70},
            color="#c0392b",
        ),
        (1, 0): RoomSpec(
            "Dining", "dining",
            objects={"vase": 0.80, "wine glass": 0.75},
            color="#d4a017",
        ),
        (2, 0): RoomSpec(
            "Living Room", "living",
            objects={"TV remote": 0.70, "cushion": 0.60, "magazine": 0.55},
            color="#2980b9",
        ),
        (3, 0): RoomSpec(
            "Balcony", "balcony",
            objects={"potted plant": 0.90},
            color="#27ae60",
        ),
        (0, 1): RoomSpec("Hallway", "hallway", color="#636e72"),
        (1, 1): RoomSpec("Entryway", "hallway", objects={"umbrella": 0.65}, color="#636e72"),
        (2, 1): RoomSpec(
            "Study", "study",
            objects={"laptop": 0.85, "notebook": 0.60},
            color="#8e44ad",
        ),
        (3, 1): RoomSpec(
            "Storage", "storage",
            objects={"toolbox": 0.70, "old phone": 0.65},
            # false positive: old broken phone detected as "phone"
            false_positives={"old phone": ("phone", 0.40)},
            color="#d35400",
        ),
        (0, 2): RoomSpec(
            "Laundry", "laundry",
            objects={"detergent": 0.75, "basket": 0.80},
            color="#00b894",
        ),
        (1, 2): RoomSpec(
            "Bathroom", "bathroom",
            objects={"towel": 0.75, "soap": 0.70},
            color="#0984e3",
        ),
        (2, 2): RoomSpec(
            "Bedroom", "bedroom",
            objects={"phone": 0.80, "lamp": 0.85, "book": 0.70},
            color="#6c5ce7",
        ),
        (3, 2): RoomSpec(
            "Guest Room", "bedroom",
            objects={"pillow": 0.75, "clock": 0.65},
            color="#e84393",
        ),
    },
)
