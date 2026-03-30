"""Spatial world built from actual 3D model mesh inventory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldObject:
    """An object discovered from parsing a GLB/GLTF scene graph."""

    mesh_name: str
    label: str
    position: tuple[float, float, float]


@dataclass
class SpatialWorld:
    """3D world populated from a real model's mesh inventory.

    Waypoints are auto-generated as a grid inside the bounding volume.
    The agent moves between waypoints and perceives nearby objects.
    """

    objects: list[WorldObject]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)
    agent_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    agent_waypoint_idx: int = 0
    visited_waypoints: set[int] = field(default_factory=set)
    step_count: int = 0

    def __post_init__(self) -> None:
        if not self.waypoints:
            self.waypoints = self._generate_waypoints()
        if self.waypoints:
            self.agent_pos = self.waypoints[0]
            self.visited_waypoints = {0}

    def _generate_waypoints(self, nx: int = 4, nz: int = 3) -> list[tuple[float, float, float]]:
        """Generate a grid of waypoints inside the bounding volume."""
        xmin, ymin, zmin = self.bounds_min
        xmax, ymax, zmax = self.bounds_max
        # Place waypoints at eye height (40% up from floor)
        eye_y = ymin + (ymax - ymin) * 0.4
        points: list[tuple[float, float, float]] = []
        for iz in range(nz):
            for ix in range(nx):
                x = xmin + (xmax - xmin) * (ix + 0.5) / nx
                z = zmin + (zmax - zmin) * (iz + 0.5) / nz
                points.append((x, eye_y, z))
        return points

    def move_to_next_unvisited(self) -> bool:
        """Move agent to the nearest unvisited waypoint. Returns False if all visited."""
        for i, wp in enumerate(self.waypoints):
            if i not in self.visited_waypoints:
                self.agent_pos = wp
                self.agent_waypoint_idx = i
                self.visited_waypoints.add(i)
                self.step_count += 1
                return True
        return False

    def move_to_waypoint(self, idx: int) -> None:
        """Move agent to a specific waypoint."""
        if 0 <= idx < len(self.waypoints):
            self.agent_pos = self.waypoints[idx]
            self.agent_waypoint_idx = idx
            self.visited_waypoints.add(idx)
            self.step_count += 1

    def revisit_nearest_to(self, target_pos: tuple[float, float, float]) -> None:
        """Move agent to the waypoint closest to a target position."""
        best_idx = 0
        best_dist = float("inf")
        for i, wp in enumerate(self.waypoints):
            d = _dist(wp, target_pos)
            if d < best_dist:
                best_dist = d
                best_idx = i
        self.move_to_waypoint(best_idx)

    @property
    def all_explored(self) -> bool:
        return len(self.visited_waypoints) >= len(self.waypoints)

    def objects_near(self, pos: tuple[float, float, float], radius: float) -> list[WorldObject]:
        """Get objects within radius of a position."""
        return [obj for obj in self.objects if _dist(pos, obj.position) <= radius]


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
