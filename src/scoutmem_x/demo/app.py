"""Gradio demo app for ScoutMem-X active scene memory."""

from __future__ import annotations

from typing import Any

import gradio as gr

from scoutmem_x.demo.renderer import render_apartment
from scoutmem_x.env.grid_world_2d import DEMO_APARTMENT, GridWorld2D, Oracle2DAdapter
from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.memory.update import build_memory_snapshot

# ── helpers ─────────────────────────────────────────────────────


def _memory_to_rows(memory: MemorySnapshot | None) -> list[list[str]]:
    if memory is None:
        return []
    rows = []
    for node in sorted(memory.nodes, key=lambda n: -n.confidence):
        state_icon = {
            "visible": "VERIFIED",
            "previously_seen": "SEEN",
            "uncertain": "UNCERTAIN",
            "hypothesized": "WEAK",
        }.get(node.visibility_state.value, node.visibility_state.value)
        rows.append([
            node.category,
            f"{node.confidence:.2f}",
            state_icon,
            str(len(node.supporting_frames)),
            node.room_or_region_estimate or "?",
        ])
    return rows


def _pack(
    world: GridWorld2D,
    memory: MemorySnapshot | None,
    log: list[str],
    adapter: Oracle2DAdapter,
    evidence: float | None = None,
    target_query: str = "",
    highlight_path: list[tuple[int, int]] | None = None,
) -> tuple[Any, ...]:
    """Build the standard return tuple for Gradio handlers."""
    img = render_apartment(world, memory, target_query, highlight_path)
    ev = evidence if evidence is not None else _evidence_score(memory)
    log_text = "\n".join(log[-20:])
    return world, memory, log, adapter, img, _memory_to_rows(memory), ev, log_text


def _evidence_score(memory: MemorySnapshot | None) -> float:
    if memory is None:
        return 0.0
    return memory.evidence_sufficiency_score


# ── core actions ────────────────────────────────────────────────


def reset_world() -> tuple[Any, ...]:
    world = GridWorld2D(DEMO_APARTMENT)
    adapter = Oracle2DAdapter()
    obs = world.reset((0, 0))
    detections = adapter.predict(obs, "")
    memory = build_memory_snapshot(obs, detections, "", None)
    room = world.current_room
    det_txt = ", ".join(f"{d.label} ({d.score:.2f})" for d in detections)
    det_msg = det_txt or "nothing"
    log = [f"[Step 0] Start in {room.name if room else '?'}  |  Detected: {det_msg}"]
    return _pack(world, memory, log, adapter)


def step_explore(
    world: GridWorld2D | None,
    memory: MemorySnapshot | None,
    log: list[str] | None,
    adapter: Oracle2DAdapter | None,
) -> tuple[Any, ...]:
    if world is None or adapter is None:
        return reset_world()
    if log is None:
        log = []

    next_pos = world.get_nearest_unvisited()
    if next_pos is None:
        log.append("[INFO] All rooms explored! Use Search to find an object.")
        return _pack(world, memory, log, adapter)

    obs = world.move_to(next_pos)
    detections = adapter.predict(obs, "")
    memory = build_memory_snapshot(obs, detections, "", memory)
    room = world.current_room
    det_txt = ", ".join(f"{d.label} ({d.score:.2f})" for d in detections)
    log.append(
        f"[Step {world.step_count}] -> {room.name if room else '?'}  |  "
        f"Detected: {det_txt or 'nothing'}"
    )
    return _pack(world, memory, log, adapter)


def auto_explore(
    world: GridWorld2D | None,
    memory: MemorySnapshot | None,
    log: list[str] | None,
    adapter: Oracle2DAdapter | None,
) -> tuple[Any, ...]:
    """Run exploration to completion."""
    if world is None or adapter is None:
        world, memory, log, adapter, *rest = reset_world()

    if log is None:
        log = []

    while not world.all_explored:
        next_pos = world.get_nearest_unvisited()
        if next_pos is None:
            break
        obs = world.move_to(next_pos)
        detections = adapter.predict(obs, "")
        memory = build_memory_snapshot(obs, detections, "", memory)
        room = world.current_room
        det_txt = ", ".join(f"{d.label} ({d.score:.2f})" for d in detections)
        log.append(
            f"[Step {world.step_count}] -> {room.name if room else '?'}  |  "
            f"Detected: {det_txt or 'nothing'}"
        )

    log.append("")
    log.append("=== EXPLORATION COMPLETE ===")
    log.append(f"Visited {len(world.visited)} rooms in {world.step_count} steps.")
    log.append(f"Memory holds {len(memory.nodes) if memory else 0} object nodes.")
    log.append('Type a query like "find the red key" and click Search.')
    return _pack(world, memory, log, adapter)


def search_object(
    world: GridWorld2D | None,
    memory: MemorySnapshot | None,
    log: list[str] | None,
    adapter: Oracle2DAdapter | None,
    query: str = "",
) -> tuple[Any, ...]:
    """Search for an object using memory, with verification."""
    if world is None or memory is None or adapter is None:
        return reset_world()
    if log is None:
        log = []

    target_label = query.strip().removeprefix("find the ").removeprefix("find ").strip()
    if not target_label:
        log.append("[WARN] Enter a search query first.")
        return _pack(world, memory, log, adapter)

    log.append("")
    log.append(f'=== SEARCH: "{target_label}" ===')

    # show all candidates in memory
    candidates = sorted(
        [n for n in memory.nodes if n.category == target_label],
        key=lambda n: -n.confidence,
    )
    if not candidates:
        log.append(f'  No "{target_label}" in memory! Try exploring first.')
        return _pack(world, memory, log, adapter, target_query=target_label)

    log.append("  Candidates in memory:")
    for i, c in enumerate(candidates):
        marker = "<- best" if i == 0 else ""
        log.append(
            f"    {c.room_or_region_estimate}: conf={c.confidence:.2f}, "
            f"frames={len(c.supporting_frames)}, "
            f"state={c.visibility_state.value}  {marker}"
        )

    # navigate + verify each candidate starting from best
    for candidate in candidates:
        if candidate.position_estimate is None:
            continue
        target_pos = (
            int(candidate.position_estimate[0]),
            int(candidate.position_estimate[1]),
        )
        path = world.find_path(target_pos)
        if not path:
            continue

        room_name = candidate.room_or_region_estimate or "?"
        log.append(f"  Navigating to {room_name} ({len(path) - 1} steps)...")

        # walk the path
        for pos in path[1:]:
            obs = world.move_to(pos)
            detections = adapter.predict(obs, target_label)
            memory = build_memory_snapshot(obs, detections, target_label, memory)

        # inspect to verify
        obs = world.inspect()
        detections = adapter.predict(obs, target_label)
        memory = build_memory_snapshot(obs, detections, target_label, memory)

        # re-check confidence
        best = retrieve_best_node(memory, target_label)
        best_conf = best.confidence if best else 0.0
        best_room = best.room_or_region_estimate if best else "?"

        if best_conf >= 0.80:
            log.append(f"  INSPECT {room_name}: confidence -> {best_conf:.2f}")
            found_msg = f"  FOUND '{target_label}' in {best_room}!"
            log.append(f"{found_msg} (conf={best_conf:.2f} >= 0.80)")
            log.append(f"  Evidence: {len(best.supporting_frames)} frames")
            return _pack(
                world, memory, log, adapter,
                evidence=best_conf,
                target_query=target_label,
                highlight_path=path,
            )
        else:
            log.append(
                f"  INSPECT {room_name}: conf {best_conf:.2f} < 0.80"
                " -- not verified, trying next..."
            )

    log.append(f'  Could not verify "{target_label}" above threshold.')
    return _pack(world, memory, log, adapter, target_query=target_label)


def search_no_memory(
    world: GridWorld2D | None,
    log: list[str] | None,
    query: str = "",
) -> str:
    """Estimate how many steps a memoryless agent would need."""
    if world is None:
        return "Reset first."
    target_label = query.strip().removeprefix("find the ").removeprefix("find ").strip()
    if not target_label:
        return ""

    # a reactive (memoryless) agent must visit every room until it finds the target
    spec = world.spec
    total_rooms = len(spec.rooms)
    target_rooms = []
    for pos, room in spec.rooms.items():
        if target_label in room.objects:
            target_rooms.append((pos, room.name))

    if not target_rooms:
        return f'No room contains "{target_label}".'

    # worst case: visit all rooms
    return (
        f"Without memory: agent must re-explore ~{total_rooms} rooms.\n"
        f'With memory: navigated directly to best candidate.\n'
        f'Rooms containing "{target_label}": '
        + ", ".join(f"{name} {pos}" for pos, name in target_rooms)
    )


# ── Gradio app ──────────────────────────────────────────────────


CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
.dark { background-color: #0d1117 !important; }
"""


def create_demo() -> gr.Blocks:
    with gr.Blocks(title="ScoutMem-X") as demo:
        gr.Markdown(
            "# ScoutMem-X  —  Active Scene Memory\n"
            "Structured memory that **reasons about certainty**, not just stores vectors.  \n"
            "Explore the apartment, then search for an object. "
            "Watch confidence build across multiple observations and false positives get rejected."
        )

        # ── state ──
        world_state = gr.State(None)
        memory_state = gr.State(None)
        log_state = gr.State([])
        adapter_state = gr.State(None)

        with gr.Row():
            # ── left: world view ──
            with gr.Column(scale=3):
                world_img = gr.Image(
                    label="Apartment", type="pil", height=480, show_label=False,
                )
                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary", size="sm")
                    step_btn = gr.Button("Step ->", size="sm")
                    explore_btn = gr.Button("Auto-Explore", variant="primary", size="sm")
                with gr.Row():
                    search_input = gr.Textbox(
                        placeholder="find the red key",
                        show_label=False,
                        scale=4,
                    )
                    search_btn = gr.Button("Search Memory", variant="primary", scale=1)
                comparison_box = gr.Textbox(
                    label="Memory Advantage", lines=3, interactive=False,
                )

            # ── right: memory + log ──
            with gr.Column(scale=2):
                gr.Markdown("### Scene Memory")
                memory_df = gr.Dataframe(
                    headers=["Object", "Confidence", "State", "Frames", "Room"],
                    interactive=False,
                    wrap=True,
                    row_count=(1, "dynamic"),
                )
                evidence_slider = gr.Slider(
                    0, 1, value=0, step=0.01,
                    label="Evidence Sufficiency (best target)",
                    interactive=False,
                )
                gr.Markdown("### Decision Log")
                decision_log = gr.Textbox(
                    lines=14, interactive=False, show_label=False,
                )

        # ── outputs tuple ──
        all_outputs = [
            world_state, memory_state, log_state, adapter_state,
            world_img, memory_df, evidence_slider, decision_log,
        ]

        reset_btn.click(reset_world, outputs=all_outputs)
        step_btn.click(
            step_explore,
            inputs=[world_state, memory_state, log_state, adapter_state],
            outputs=all_outputs,
        )
        explore_btn.click(
            auto_explore,
            inputs=[world_state, memory_state, log_state, adapter_state],
            outputs=all_outputs,
        )
        search_btn.click(
            search_object,
            inputs=[world_state, memory_state, log_state, adapter_state, search_input],
            outputs=all_outputs,
        )
        # comparison text after search
        search_btn.click(
            search_no_memory,
            inputs=[world_state, log_state, search_input],
            outputs=[comparison_box],
        )
        # auto-reset on load
        demo.load(reset_world, outputs=all_outputs)

    return demo


# ── entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_demo()
    app.launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="amber", neutral_hue="gray"),
        css=CUSTOM_CSS,
    )
