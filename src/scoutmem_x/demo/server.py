"""FastAPI server for the ScoutMem-X 3D demo.

The world is populated from actual GLB mesh data sent by the frontend.
Perception is noisy and distance-based — the agent must explore to build
confidence, exactly like a real embodied agent with imperfect vision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.memory.update import build_memory_snapshot
from scoutmem_x.spatial.perception import SpatialPerceptionAdapter
from scoutmem_x.spatial.world import SpatialWorld, WorldObject

app = FastAPI(title="ScoutMem-X 3D Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── global state ────────────────────────────────────────────────
_world: SpatialWorld | None = None
_adapter: SpatialPerceptionAdapter | None = None
_memory: MemorySnapshot | None = None
_log: list[str] = []


def _build_observation(step: int = 0) -> Observation:
    """Build an Observation from the current spatial world state."""
    assert _world is not None
    pos = _world.agent_pos
    return Observation(
        frame_id=f"spatial-step-{step}",
        step_index=step,
        pose=pos,
        heading_radians=0.0,
        image_size=(256, 256),
        metadata={
            "scene_id": "spatial",
            "agent_x": str(pos[0]),
            "agent_y": str(pos[1]),
            "agent_z": str(pos[2]),
            "waypoint": str(_world.agent_waypoint_idx),
        },
    )


def _memory_objects() -> list[dict[str, Any]]:
    """Extract all memory nodes as dicts for the API response."""
    if _memory is None:
        return []
    result = []
    for node in sorted(_memory.nodes, key=lambda n: -n.confidence):
        result.append({
            "label": node.category,
            "confidence": round(node.confidence, 3),
            "state": node.visibility_state.value,
            "frames": len(node.supporting_frames),
            "region": node.room_or_region_estimate or "?",
            "position": list(node.position_estimate) if node.position_estimate else None,
        })
    return result


def _state_dict() -> dict[str, Any]:
    return {
        "objects": _memory_objects(),
        "agent": list(_world.agent_pos) if _world else [0, 0, 0],
        "step": _world.step_count if _world else 0,
        "allExplored": _world.all_explored if _world else False,
        "memoryCount": len(_memory.nodes) if _memory else 0,
        "evidence": round(_memory.evidence_sufficiency_score, 3) if _memory else 0,
        "waypointsTotal": len(_world.waypoints) if _world else 0,
        "waypointsVisited": len(_world.visited_waypoints) if _world else 0,
        "log": _log[-30:],
    }


# ── routes ──────────────────────────────────────────────────────


class InitWorldPayload(BaseModel):
    objects: list[dict[str, Any]]
    bounds_min: list[float]
    bounds_max: list[float]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/init-world")
def init_world(payload: InitWorldPayload) -> dict[str, Any]:
    """Initialize the spatial world from GLB mesh inventory.

    Called by the frontend after it parses the 3D model.
    """
    global _world, _adapter, _memory, _log

    world_objects = [
        WorldObject(
            mesh_name=o.get("name", ""),
            label=o.get("cleanName", o.get("name", "")),
            position=(
                o["position"][0],
                o["position"][1],
                o["position"][2],
            ),
        )
        for o in payload.objects
        if len(o.get("cleanName", "")) > 1
    ]

    bmin = tuple(payload.bounds_min)
    bmax = tuple(payload.bounds_max)

    _world = SpatialWorld(
        objects=world_objects,
        bounds_min=(bmin[0], bmin[1], bmin[2]),
        bounds_max=(bmax[0], bmax[1], bmax[2]),
    )
    _adapter = SpatialPerceptionAdapter(_world, view_range=8.0, dropout_rate=0.15)
    _memory = None
    _log = [
        f"World initialized: {len(world_objects)} objects, "
        f"{len(_world.waypoints)} waypoints",
    ]

    return {
        "status": "ok",
        "objectCount": len(world_objects),
        "waypointCount": len(_world.waypoints),
    }


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    return _state_dict()


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    global _memory, _log
    if _world is not None:
        _world.agent_pos = _world.waypoints[0] if _world.waypoints else (0, 0, 0)
        _world.agent_waypoint_idx = 0
        _world.visited_waypoints = {0}
        _world.step_count = 0
    _memory = None
    _log = ["Agent reset to starting position."]
    return _state_dict()


@app.post("/api/step")
def step() -> dict[str, Any]:
    """Move agent to the next unvisited waypoint and perceive."""
    global _memory
    if _world is None or _adapter is None:
        return {**_state_dict(), "done": True, "message": "Init world first"}

    moved = _world.move_to_next_unvisited()
    if not moved:
        _log.append("All waypoints visited!")
        return {**_state_dict(), "done": True}

    obs = _build_observation(_world.step_count)
    dets = _adapter.predict(obs, "")
    _memory = build_memory_snapshot(obs, dets, "", _memory)

    det_txt = ", ".join(f"{d.label} ({d.score:.2f})" for d in dets)
    wp = _world.agent_waypoint_idx
    _log.append(
        f"Step {_world.step_count}: waypoint {wp}"
        f"  |  Detected {len(dets)} objects"
        f"  |  {det_txt[:80] or 'nothing nearby'}"
    )
    return {**_state_dict(), "done": False}


@app.post("/api/explore")
def auto_explore() -> dict[str, Any]:
    """Visit all waypoints, building memory from noisy perception."""
    global _memory
    if _world is None or _adapter is None:
        return {**_state_dict(), "message": "Init world first"}

    while not _world.all_explored:
        moved = _world.move_to_next_unvisited()
        if not moved:
            break
        obs = _build_observation(_world.step_count)
        dets = _adapter.predict(obs, "")
        _memory = build_memory_snapshot(obs, dets, "", _memory)

        det_txt = ", ".join(f"{d.label} ({d.score:.2f})" for d in dets[:5])
        _log.append(
            f"Step {_world.step_count}: waypoint {_world.agent_waypoint_idx}"
            f"  |  {len(dets)} detections  |  {det_txt[:60]}"
        )

    node_count = len(_memory.nodes) if _memory else 0
    _log.append(
        f"Exploration complete: {_world.step_count} steps, "
        f"{node_count} unique objects in memory"
    )
    return _state_dict()


@app.post("/api/search")
def search(q: str = Query(...)) -> dict[str, Any]:
    """Search memory for an object, re-verify by revisiting nearby waypoints."""
    global _memory
    if _world is None or _adapter is None or _memory is None:
        return {**_state_dict(), "found": False, "message": "Explore first"}

    target = q.strip().removeprefix("find the ").removeprefix("find ").strip()
    if not target:
        return {**_state_dict(), "found": False, "message": "Empty query"}

    # Find all memory nodes matching the query
    candidates = sorted(
        [n for n in _memory.nodes if target.lower() in n.category.lower()],
        key=lambda n: -n.confidence,
    )

    if not candidates:
        _log.append(f'Search "{target}": not found in memory')
        return {**_state_dict(), "found": False, "message": f'"{target}" not in memory'}

    _log.append(f'Search "{target}": {len(candidates)} candidate(s)')
    for c in candidates:
        _log.append(
            f"  {c.room_or_region_estimate}: conf={c.confidence:.2f}, "
            f"frames={len(c.supporting_frames)}"
        )

    # Navigate to best candidate and re-verify
    best = candidates[0]
    if best.position_estimate:
        _world.revisit_nearest_to(best.position_estimate)
        obs = _build_observation(_world.step_count)
        dets = _adapter.predict(obs, target)
        _memory = build_memory_snapshot(obs, dets, target, _memory)

        # Re-check after verification
        best = retrieve_best_node(_memory, target) or best

    found = best.confidence >= 0.6  # lower threshold since perception is noisy
    pos = list(best.position_estimate) if best.position_estimate else None

    if found:
        _log.append(
            f'FOUND "{target}" at {best.room_or_region_estimate} '
            f"(conf={best.confidence:.2f}, {len(best.supporting_frames)} frames)"
        )
    else:
        _log.append(
            f'"{target}" below confidence threshold '
            f"(conf={best.confidence:.2f})"
        )

    return {
        **_state_dict(),
        "found": found,
        "confidence": round(best.confidence, 3),
        "region": best.room_or_region_estimate,
        "targetPosition": pos,
    }


# ── entrypoint ──────────────────────────────────────────────────

def main() -> None:
    import uvicorn
    print("\n  ScoutMem-X 3D Demo")
    print("  Open http://localhost:7860 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")


if __name__ == "__main__":
    main()
