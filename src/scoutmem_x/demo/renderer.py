"""Visual renderer for the ScoutMem-X 2D apartment demo."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from PIL import Image

if TYPE_CHECKING:
    from scoutmem_x.env.grid_world_2d import GridWorld2D
    from scoutmem_x.memory.schema import MemorySnapshot

matplotlib.use("Agg")

# ── colour palette ──────────────────────────────────────────────
BG = "#0d1117"
ROOM_BORDER = "#30363d"
AGENT_COLOR = "#f0c040"
AGENT_RING = "#ffffff"
TEXT_WHITE = "#e6edf3"
TEXT_DIM = "#8b949e"
CONF_HIGH = "#3fb950"
CONF_MED = "#d29922"
CONF_LOW = "#f85149"
PATH_GLOW = "#f0c04030"


def _conf_color(conf: float) -> str:
    if conf >= 0.8:
        return CONF_HIGH
    if conf >= 0.5:
        return CONF_MED
    return CONF_LOW


def render_apartment(
    world: GridWorld2D,
    memory: MemorySnapshot | None = None,
    target_query: str = "",
    highlight_path: list[tuple[int, int]] | None = None,
) -> Image.Image:
    """Render a top-down apartment view as a PIL image."""
    spec = world.spec
    cw, ch = 3.2, 2.6  # cell size
    fig_w = spec.width * cw + 0.6
    fig_h = spec.height * ch + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # index memory confidence by (room_name, category)
    mem_lookup: dict[tuple[str, str], tuple[float, str]] = {}
    if memory:
        for node in memory.nodes:
            key = (node.room_or_region_estimate or "", node.category)
            mem_lookup[key] = (node.confidence, node.visibility_state.value)

    path_set = set(highlight_path) if highlight_path else set()

    for (x, y), room in spec.rooms.items():
        rx = x * cw + 0.25
        ry = (spec.height - 1 - y) * ch + 0.25
        w = cw - 0.5
        h = ch - 0.5

        # background tint
        alpha = 0.55 if (x, y) in world.visited else 0.20
        facecolor = to_rgba(room.color, alpha)
        if (x, y) in path_set:
            facecolor = to_rgba(AGENT_COLOR, 0.18)

        rect = patches.FancyBboxPatch(
            (rx, ry), w, h,
            boxstyle="round,pad=0.08",
            facecolor=facecolor,
            edgecolor=ROOM_BORDER,
            linewidth=1.4,
        )
        ax.add_patch(rect)

        # room name
        ax.text(
            rx + 0.15, ry + h - 0.22,
            room.name.upper(),
            color=TEXT_WHITE, fontsize=7.5, fontweight="bold",
            fontfamily="monospace", va="top",
        )

        # objects with optional confidence overlay
        obj_y = ry + h - 0.60
        all_labels = list(room.objects.keys())
        for _fp_real, (fp_label, _) in room.false_positives.items():
            if fp_label not in all_labels:
                all_labels.append(fp_label)
        for label in all_labels:
            conf_info = mem_lookup.get((room.name, label))
            if conf_info is not None:
                conf, _state = conf_info
                color = _conf_color(conf)
                txt = f"{label}  [{conf:.2f}]"
            elif (x, y) in world.visited:
                color = TEXT_DIM
                txt = label
            else:
                color = TEXT_DIM + "60"  # very faint for unvisited
                txt = label
            ax.text(
                rx + 0.22, obj_y, txt,
                color=color, fontsize=6, fontfamily="monospace",
            )
            obj_y -= 0.32

    # agent marker
    ax_pos, ay_pos = world.agent_pos
    cx = ax_pos * cw + cw / 2
    cy = (spec.height - 1 - ay_pos) * ch + 0.55
    glow = patches.Circle(
        (cx, cy), 0.35, facecolor=AGENT_COLOR + "30", edgecolor="none", zorder=9,
    )
    ax.add_patch(glow)
    dot = patches.Circle(
        (cx, cy), 0.22, facecolor=AGENT_COLOR, edgecolor=AGENT_RING,
        linewidth=2.0, zorder=10,
    )
    ax.add_patch(dot)
    ax.text(
        cx, cy, "A", ha="center", va="center",
        color="#000000", fontsize=9, fontweight="bold", zorder=11,
    )

    # title
    room_label = world.current_room.name if world.current_room else "?"
    title = f"Step {world.step_count}  ·  Agent in {room_label}"
    if world.all_explored:
        title += "  ·  All rooms explored"
    ax.text(
        fig_w / 2, spec.height * ch + 0.55, title,
        ha="center", va="center", color=TEXT_WHITE,
        fontsize=10, fontweight="bold", fontfamily="monospace",
    )

    ax.set_xlim(0, spec.width * cw + 0.1)
    ax.set_ylim(0, spec.height * ch + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.2)
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=130, bbox_inches="tight",
        facecolor=BG, edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
