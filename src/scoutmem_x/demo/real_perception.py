"""Real perception demo: GroundingDINO + ScoutMem-X memory on actual images.

Demonstrates the core advantage of ScoutMem-X over vector DB retrieval:
multiple noisy observations are aggregated into confident beliefs.

Usage:
    # Run on a folder of images (simulating agent looking around a room)
    python -m scoutmem_x.demo.real_perception --images path/to/images/ --target "mug"

    # Run on a single image (shows single-shot vs multi-observation)
    python -m scoutmem_x.demo.real_perception --images photo.jpg --target "laptop"

    # Generate sample images for testing (downloads from web)
    python -m scoutmem_x.demo.real_perception --generate-samples --target "cup"

Requires: pip install -e ".[perception]"  (transformers, torch, pillow)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoutmem_x.env.observation import Observation
from scoutmem_x.memory.retrieval import retrieve_best_node
from scoutmem_x.memory.schema import MemorySnapshot
from scoutmem_x.memory.update import build_memory_snapshot
from scoutmem_x.perception.adapters import Detection


@dataclass
class PerceptionResult:
    image_path: str
    detections: list[Detection]
    memory_after: MemorySnapshot
    target_confidence: float
    n_candidates: int


def run_real_perception(
    image_paths: list[Path],
    target_label: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
    decay_rate: float = 0.01,
) -> list[PerceptionResult]:
    """Run GroundingDINO on each image, aggregate into ScoutMem-X memory."""
    from scoutmem_x.perception.grounding_dino import GroundingDINOAdapter

    adapter = GroundingDINOAdapter(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    memory: MemorySnapshot | None = None
    results: list[PerceptionResult] = []

    for i, img_path in enumerate(image_paths):
        # Create observation for this "step"
        obs = Observation(
            frame_id=f"real-{i}",
            step_index=i,
            pose=(0.0, 0.0, 0.0),
            heading_radians=0.0,
            image_size=(640, 480),
            rgb_path=str(img_path),
        )

        # Run GroundingDINO
        query = f"{target_label}."
        detections = adapter.predict(obs, query)

        # Feed into ScoutMem-X memory
        memory = build_memory_snapshot(
            obs, detections, target_label, memory, decay_rate=decay_rate,
        )

        # Get current target confidence
        best = retrieve_best_node(memory, target_label) if memory else None
        target_conf = best.confidence if best else 0.0
        n_candidates = sum(
            1 for n in memory.nodes if n.category == target_label
        ) if memory else 0

        results.append(PerceptionResult(
            image_path=str(img_path),
            detections=detections,
            memory_after=memory,
            target_confidence=target_conf,
            n_candidates=n_candidates,
        ))

    return results


def run_vectordb_baseline(
    image_paths: list[Path],
    target_label: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.20,
) -> float:
    """Simulate vector DB approach: store best single detection, return its score."""
    from scoutmem_x.perception.grounding_dino import GroundingDINOAdapter

    adapter = GroundingDINOAdapter(
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    best_score = 0.0
    for img_path in image_paths:
        obs = Observation(
            frame_id="vdb",
            step_index=0,
            pose=(0.0, 0.0, 0.0),
            heading_radians=0.0,
            image_size=(640, 480),
            rgb_path=str(img_path),
        )
        detections = adapter.predict(obs, f"{target_label}.")
        for d in detections:
            if target_label.lower() in d.label.lower():
                best_score = max(best_score, d.score)

    return best_score


def print_results(results: list[PerceptionResult], target: str, vdb_score: float) -> None:
    """Print a comparison table."""
    print(f"\n{'='*70}")
    print(f"ScoutMem-X Real Perception Demo")
    print(f"Target: '{target}' | {len(results)} images processed")
    print(f"{'='*70}")

    print(f"\n{'Step':<6} {'Image':<30} {'Detections':<12} {'ScoutMem Conf':<15}")
    print(f"{'-'*70}")
    for i, r in enumerate(results):
        n_det = len(r.detections)
        det_str = f"{n_det} found" if n_det > 0 else "none"
        conf_bar = "#" * int(r.target_confidence * 20)
        print(
            f"{i:<6} {Path(r.image_path).name:<30} {det_str:<12} "
            f"{r.target_confidence:.3f} |{conf_bar}"
        )

    final_conf = results[-1].target_confidence if results else 0.0
    print(f"\n{'='*70}")
    print(f"COMPARISON")
    print(f"{'='*70}")
    print(f"  Vector DB (best single score):  {vdb_score:.3f}")
    print(f"  ScoutMem-X (aggregated):        {final_conf:.3f}")
    if final_conf > vdb_score:
        delta = final_conf - vdb_score
        print(f"  ScoutMem-X advantage:           +{delta:.3f} ({delta/max(vdb_score,0.001):.0%} improvement)")
    print(f"{'='*70}")


def save_results(
    results: list[PerceptionResult], target: str, vdb_score: float, output_path: Path,
) -> None:
    """Save results as JSON."""
    data = {
        "target": target,
        "n_images": len(results),
        "vector_db_best_score": vdb_score,
        "scoutmem_final_confidence": results[-1].target_confidence if results else 0.0,
        "per_step": [
            {
                "image": r.image_path,
                "n_detections": len(r.detections),
                "detection_scores": [d.score for d in r.detections],
                "scoutmem_confidence": r.target_confidence,
                "n_candidates": r.n_candidates,
            }
            for r in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real perception demo: GroundingDINO + ScoutMem-X memory"
    )
    parser.add_argument(
        "--images", type=str, required=True,
        help="Path to image file or directory of images",
    )
    parser.add_argument("--target", type=str, required=True, help="Target object to find")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--output", type=str, default="outputs/real_perception/results.json")
    args = parser.parse_args()

    # Collect image paths
    img_path = Path(args.images)
    if img_path.is_dir():
        image_paths = sorted(
            p for p in img_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    elif img_path.is_file():
        image_paths = [img_path]
    else:
        print(f"Error: {img_path} is not a file or directory")
        return

    if not image_paths:
        print(f"No images found in {img_path}")
        return

    print(f"Found {len(image_paths)} images")
    print(f"Target: '{args.target}'")
    print(f"Loading GroundingDINO (first run downloads the model)...\n")

    # Run ScoutMem-X pipeline
    results = run_real_perception(
        image_paths, args.target,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # Run vector DB baseline
    vdb_score = run_vectordb_baseline(
        image_paths, args.target,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    # Output
    print_results(results, args.target, vdb_score)
    save_results(results, args.target, vdb_score, Path(args.output))


if __name__ == "__main__":
    main()
