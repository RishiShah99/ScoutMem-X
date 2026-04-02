from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.memory.update import build_memory_snapshot
from scoutmem_x.perception.adapters import Detection


def run_pipeline(
    image_paths: list[Path], target: str, box_thresh: float, text_thresh: float,
) -> list[dict[str, Any]]:
    """Run GroundingDINO + ScoutMem on each image, collect all data."""
    from scoutmem_x.perception.grounding_dino import GroundingDINOAdapter

    adapter = GroundingDINOAdapter(
        box_threshold=box_thresh, text_threshold=text_thresh,
    )

    memory: MemorySnapshot | None = None
    vdb_best = 0.0
    steps: list[dict[str, Any]] = []

    for i, img_path in enumerate(image_paths):
        obs = Observation(
            frame_id=f"real-{i}", step_index=i,
            pose=(0.0, 0.0, 0.0), heading_radians=0.0,
            image_size=(640, 480), rgb_path=str(img_path),
        )

        detections = adapter.predict(obs, f"{target}.")

        memory = build_memory_snapshot(
            obs, detections, target, memory, decay_rate=0.01,
        )

        best = retrieve_best_node(memory, target) if memory else None
        scoutmem_conf = best.confidence if best else 0.0

        for d in detections:
            if target.lower() in d.label.lower():
                vdb_best = max(vdb_best, d.score)

        steps.append({
            "image_path": str(img_path),
            "detections": detections,
            "scoutmem_conf": scoutmem_conf,
            "vdb_best": vdb_best,
        })

    return steps


def generate_comparison_figure(
    steps: list[dict[str, Any]], target: str, output_path: Path,
) -> None:
    """Create the visual comparison figure."""
    n = len(steps)

    fig = plt.figure(figsize=(16, 10))

    # Top row: images with bounding boxes (takes 70% of height)
    # Bottom: confidence comparison chart (takes 30%)
    gs = fig.add_gridspec(2, n, height_ratios=[2.5, 1], hspace=0.35, wspace=0.15)

    # -- Top: images with detections --
    for i, step in enumerate(steps):
        ax = fig.add_subplot(gs[0, i])
        img = Image.open(step["image_path"])
        ax.imshow(img)

        # Draw bounding boxes
        for det in step["detections"]:
            if det.region:
                x1, y1, x2, y2 = det.region
                rect = mpatches.FancyBboxPatch(
                    (x1, y1), x2 - x1, y2 - y1,
                    boxstyle="round,pad=2",
                    linewidth=2, edgecolor="#22c55e", facecolor="none",
                )
                ax.add_patch(rect)
                label_text = f"{det.label} {det.score:.0%}"
                ax.text(
                    x1, y1 - 5, label_text,
                    fontsize=8, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#22c55e", alpha=0.85),
                )

        # Confidence badge
        conf = step["scoutmem_conf"]
        badge_color = "#ef4444" if conf < 0.5 else ("#f59e0b" if conf < 0.8 else "#22c55e")
        ax.text(
            0.98, 0.02, f"ScoutMem: {conf:.0%}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="white", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_color, alpha=0.9),
        )

        ax.set_title(f"Observation {i + 1}", fontsize=11, fontweight="bold")
        ax.axis("off")

    # -- Bottom: confidence comparison chart --
    ax_chart = fig.add_subplot(gs[1, :])

    x = list(range(1, n + 1))
    scoutmem_confs = [s["scoutmem_conf"] for s in steps]
    vdb_confs = [s["vdb_best"] for s in steps]

    ax_chart.plot(x, scoutmem_confs, "o-", color="#1e40af", linewidth=3,
                  markersize=10, label="ScoutMem-X (Bayesian aggregation)", zorder=5)
    ax_chart.plot(x, vdb_confs, "s--", color="#9ca3af", linewidth=2.5,
                  markersize=8, label="Vector DB (best single score)", zorder=4)

    # Fill area between them
    ax_chart.fill_between(x, vdb_confs, scoutmem_confs,
                          color="#1e40af", alpha=0.08)

    # Annotate final values
    ax_chart.annotate(
        f"{scoutmem_confs[-1]:.0%}",
        (x[-1], scoutmem_confs[-1]),
        textcoords="offset points", xytext=(12, -5),
        fontsize=13, fontweight="bold", color="#1e40af",
    )
    ax_chart.annotate(
        f"{vdb_confs[-1]:.0%}",
        (x[-1], vdb_confs[-1]),
        textcoords="offset points", xytext=(12, 5),
        fontsize=13, fontweight="bold", color="#9ca3af",
    )

    # Evidence threshold line
    ax_chart.axhline(y=0.8, color="#22c55e", linestyle=":", linewidth=1.5,
                     alpha=0.6, label="Evidence threshold (80%)")

    ax_chart.set_xlabel("Observation #", fontsize=12)
    ax_chart.set_ylabel("Confidence", fontsize=12)
    ax_chart.set_ylim(0, 1.08)
    ax_chart.set_xlim(0.5, n + 0.5)
    ax_chart.set_xticks(x)
    ax_chart.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax_chart.grid(True, alpha=0.2)
    ax_chart.spines["top"].set_visible(False)
    ax_chart.spines["right"].set_visible(False)

    # Gap annotation
    gap = scoutmem_confs[-1] - vdb_confs[-1]
    if gap > 0.01:
        mid_y = (scoutmem_confs[-1] + vdb_confs[-1]) / 2
        ax_chart.annotate(
            f"+{gap:.0%}\nimprovement",
            xy=(n + 0.3, mid_y), fontsize=10, fontweight="bold",
            color="#1e40af", ha="left", va="center",
        )

    fig.suptitle(
        f'ScoutMem-X vs Vector DB: Finding "{target}" Across Multiple Observations',
        fontsize=15, fontweight="bold", y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--output", type=str, default="figures/real_perception_comparison.png")
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()

    img_path = Path(args.images)
    if img_path.is_dir():
        image_paths = sorted(
            p for p in img_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )[:args.max_images]
    else:
        image_paths = [img_path]

    print(f"Processing {len(image_paths)} images, target: '{args.target}'")

    steps = run_pipeline(image_paths, args.target, args.box_threshold, args.text_threshold)
    generate_comparison_figure(steps, args.target, Path(args.output))


if __name__ == "__main__":
    main()
