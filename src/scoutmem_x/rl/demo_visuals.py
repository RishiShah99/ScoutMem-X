"""Generate visual demo images for the README showing the environment and agent behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Object icons (emoji-like labels for grid)
OBJ_ICONS = {
    "mug": "MUG", "book": "BOOK", "phone": "PHONE", "key": "KEY",
    "lamp": "LAMP", "plant": "PLANT", "remote": "RMT", "bottle": "BTL",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
    })


def draw_grid_env(
    grid_size: int,
    agent_pos: tuple[int, int],
    objects: list[dict],
    target_idx: int,
    visited: set[tuple[int, int]],
    view_range: float,
    title: str,
    ax: plt.Axes,
) -> None:
    """Draw a single grid environment state."""
    # Draw grid cells
    for r in range(grid_size):
        for c in range(grid_size):
            color = "#e8f4e8" if (r, c) in visited else "#ffffff"
            rect = mpatches.FancyBboxPatch(
                (c - 0.45, r - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05", facecolor=color,
                edgecolor="#d0d0d0", linewidth=1,
            )
            ax.add_patch(rect)

    # Draw view range circle
    circle = plt.Circle(
        (agent_pos[1], agent_pos[0]), view_range,
        fill=True, facecolor="#3b82f620", edgecolor="#3b82f6",
        linewidth=1.5, linestyle="--",
    )
    ax.add_patch(circle)

    # Draw objects
    for i, obj in enumerate(objects):
        r, c = obj["pos"]
        is_target = i == target_idx
        is_distractor = obj.get("is_distractor", False)

        if is_target:
            color = "#ef4444"
            marker_color = "#ef4444"
        elif is_distractor:
            color = "#f59e0b"
            marker_color = "#f59e0b"
        else:
            color = "#6b7280"
            marker_color = "#6b7280"

        ax.plot(c, r, "o", color=marker_color, markersize=18, zorder=5)
        short = OBJ_ICONS.get(obj["label"], obj["label"][:3].upper())
        ax.text(c, r, short, ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=6)

    # Draw agent
    ax.plot(agent_pos[1], agent_pos[0], "s", color="#1e40af",
            markersize=22, zorder=10)
    ax.text(agent_pos[1], agent_pos[0], "A", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=11)

    ax.set_xlim(-0.6, grid_size - 0.4)
    ax.set_ylim(grid_size - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)


def generate_difficulty_comparison(output_dir: Path) -> None:
    """Side-by-side: easy vs medium vs hard environments."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Easy: 3x3, 3 objects, 0 distractors
    easy_objects = [
        {"label": "mug", "pos": (0, 2), "is_distractor": False},
        {"label": "book", "pos": (2, 0), "is_distractor": False},
        {"label": "phone", "pos": (1, 1), "is_distractor": False},
    ]
    draw_grid_env(
        3, (1, 0), easy_objects, 0, {(1, 0), (0, 0)}, 2.0,
        "EASY (3x3)\n3 objects, 0 distractors\n5% dropout, low noise",
        axes[0],
    )

    # Medium: 4x4, 5 objects, 1 distractor
    med_objects = [
        {"label": "mug", "pos": (0, 3), "is_distractor": False},
        {"label": "book", "pos": (3, 0), "is_distractor": False},
        {"label": "phone", "pos": (1, 1), "is_distractor": False},
        {"label": "key", "pos": (2, 2), "is_distractor": False},
        {"label": "lamp", "pos": (3, 3), "is_distractor": False},
        {"label": "mug", "pos": (2, 0), "is_distractor": True},
    ]
    draw_grid_env(
        4, (2, 1), med_objects, 0, {(2, 1), (1, 1), (2, 2)}, 2.0,
        "MEDIUM (4x4)\n5 objects, 1 distractor\n8% dropout, moderate noise",
        axes[1],
    )

    # Hard: 5x5, 6 objects, 2 distractors
    hard_objects = [
        {"label": "mug", "pos": (0, 4), "is_distractor": False},
        {"label": "book", "pos": (4, 0), "is_distractor": False},
        {"label": "phone", "pos": (1, 2), "is_distractor": False},
        {"label": "key", "pos": (3, 1), "is_distractor": False},
        {"label": "lamp", "pos": (2, 3), "is_distractor": False},
        {"label": "plant", "pos": (4, 4), "is_distractor": False},
        {"label": "mug", "pos": (3, 3), "is_distractor": True},
        {"label": "mug", "pos": (1, 0), "is_distractor": True},
    ]
    draw_grid_env(
        5, (2, 2), hard_objects, 0, {(2, 2), (1, 2), (2, 3)}, 2.0,
        "HARD (5x5)\n6 objects, 2 distractors\n10% dropout, high noise",
        axes[2],
    )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#1e40af", label="Agent"),
        mpatches.Patch(facecolor="#ef4444", label="Target"),
        mpatches.Patch(facecolor="#f59e0b", label="Distractor (same label)"),
        mpatches.Patch(facecolor="#6b7280", label="Other objects"),
        mpatches.Patch(facecolor="#e8f4e8", edgecolor="#d0d0d0", label="Visited cells"),
        mpatches.Patch(facecolor="#3b82f620", edgecolor="#3b82f6", label="View range"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=6,
        fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("ScoutMem-X Environment Difficulty Levels", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "env_difficulties.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved env_difficulties.png")


def generate_episode_trace(output_dir: Path) -> None:
    """Show a 6-step episode with the agent searching and confidence building."""
    _style()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Simulated episode on a 4x4 grid
    grid = 4
    target_pos = (0, 3)
    objects = [
        {"label": "mug", "pos": (0, 3), "is_distractor": False},
        {"label": "book", "pos": (3, 0), "is_distractor": False},
        {"label": "phone", "pos": (2, 1), "is_distractor": False},
        {"label": "key", "pos": (1, 2), "is_distractor": False},
        {"label": "mug", "pos": (3, 2), "is_distractor": True},
    ]

    # Episode steps: position, visited set, confidence, action taken
    steps = [
        ((2, 0), {(2, 0)}, 0.0, "Start: exploring"),
        ((1, 0), {(2, 0), (1, 0)}, 0.0, "Move up, nothing found"),
        ((1, 1), {(2, 0), (1, 0), (1, 1)}, 0.0, "Move right, sees phone"),
        ((1, 2), {(2, 0), (1, 0), (1, 1), (1, 2)}, 0.32, "Sees 'mug' (distractor)"),
        ((0, 2), {(2, 0), (1, 0), (1, 1), (1, 2), (0, 2)}, 0.51, "Moves toward signal"),
        ((0, 3), {(2, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 3)}, 0.78, "Sees real mug!"),
        ((0, 3), {(2, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 3)}, 0.91, "Re-observe: conf 0.91"),
        ((0, 3), {(2, 0), (1, 0), (1, 1), (1, 2), (0, 2), (0, 3)}, 0.91, "STOP (conf > 0.8)"),
    ]

    for i, (pos, visited, conf, desc) in enumerate(steps):
        ax = axes[i // 4][i % 4]
        draw_grid_env(grid, pos, objects, 0, visited, 2.0, "", ax)

        # Confidence bar at bottom
        bar_color = "#ef4444" if conf < 0.5 else ("#f59e0b" if conf < 0.8 else "#22c55e")
        ax.barh(-0.9, conf * (grid - 1), height=0.25, color=bar_color, left=-0.4, zorder=15)
        ax.barh(-0.9, (grid - 1), height=0.25, color="#e5e7eb", left=-0.4, zorder=14)
        ax.text(grid / 2 - 0.4, -0.9, f"Conf: {conf:.0%}", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=16)

        step_label = f"Step {i}" if i < 7 else "FOUND!"
        ax.set_title(f"{step_label}\n{desc}", fontsize=9, pad=5)
        ax.set_ylim(grid - 0.2, -1.2)

    fig.suptitle("Episode Trace: Agent Searching for 'mug' (Medium Difficulty)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_dir / "episode_trace.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved episode_trace.png")


def generate_confidence_buildup(output_dir: Path) -> None:
    """Show Bayesian confidence accumulation vs vector DB single-shot."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 4))

    steps = range(6)
    detections = [0.35, 0.42, 0.38, 0.45, 0.40, 0.43]

    # ScoutMem-X: Bayesian aggregation
    scoutmem_conf = []
    conf = 0.0
    for d in detections:
        conf = 1 - (1 - conf) * (1 - d)
        scoutmem_conf.append(conf)

    # Vector DB: just keeps the best single score
    vectordb_conf = []
    best = 0.0
    for d in detections:
        best = max(best, d)
        vectordb_conf.append(best)

    ax.plot(steps, scoutmem_conf, "o-", color="#1e40af", linewidth=2.5,
            markersize=10, label="ScoutMem-X (Bayesian)", zorder=5)
    ax.plot(steps, vectordb_conf, "s--", color="#9ca3af", linewidth=2,
            markersize=8, label="Vector DB (best single)", zorder=4)

    # Annotate individual detections
    for i, d in enumerate(detections):
        ax.annotate(f"det={d:.2f}", (i, scoutmem_conf[i]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=7, color="#1e40af", alpha=0.7)

    ax.axhline(y=0.8, color="#22c55e", linestyle=":", linewidth=1.5,
               label="Evidence threshold (0.8)", alpha=0.7)
    ax.fill_between(steps, 0.8, 1.0, color="#22c55e", alpha=0.05)

    ax.set_xlabel("Observation #")
    ax.set_ylabel("Confidence")
    ax.set_title("Bayesian Confidence Aggregation vs Single-Shot Retrieval",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.3, 5.3)
    ax.grid(True, alpha=0.3)

    # Add formula annotation
    ax.text(0.02, 0.95, "Formula: conf = 1 - (1 - prior) * (1 - score)",
            transform=ax.transAxes, fontsize=8, color="#6b7280",
            fontstyle="italic", verticalalignment="top")

    fig.tight_layout()
    fig.savefig(str(output_dir / "confidence_buildup.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved confidence_buildup.png")


def generate_architecture_diagram(output_dir: Path) -> None:
    """Pipeline architecture diagram."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 3))

    boxes = [
        ("Agent\nMoves", "#3b82f6"),
        ("Perception\n(noisy detections)", "#ef4444"),
        ("Memory Update\n(Bayesian aggregation)", "#f59e0b"),
        ("Policy Decision\n(PPO + RND)", "#8b5cf6"),
        ("Action\n(move/stop)", "#22c55e"),
    ]

    for i, (label, color) in enumerate(boxes):
        x = i * 2.2
        rect = mpatches.FancyBboxPatch(
            (x, 0.2), 1.8, 1.6,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
            linewidth=2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x + 0.9, 1.0, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

        if i < len(boxes) - 1:
            ax.annotate("", xy=((i + 1) * 2.2, 1.0), xytext=(x + 1.8, 1.0),
                        arrowprops=dict(arrowstyle="->", color="#374151", lw=2))

    # Loop arrow from last to first
    ax.annotate("", xy=(0, 0.2), xytext=(4 * 2.2 + 1.8, 0.2),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(4.5, -0.3, "repeat each step", ha="center", fontsize=9,
            color="#6b7280", fontstyle="italic")

    ax.set_xlim(-0.3, 10.8)
    ax.set_ylim(-0.8, 2.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("ScoutMem-X Pipeline: Each Step", fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(str(output_dir / "architecture.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved architecture.png")


def main() -> None:
    out = Path("figures")
    out.mkdir(exist_ok=True)

    print("Generating demo visuals...")
    generate_difficulty_comparison(out)
    generate_episode_trace(out)
    generate_confidence_buildup(out)
    generate_architecture_diagram(out)
    print(f"\nAll demo visuals saved to {out}/")


if __name__ == "__main__":
    main()
