from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scoutmem_x.config import AppConfig
from scoutmem_x.env import load_default_scenes
from scoutmem_x.eval import compare_baselines, evaluate_passive_memory_baseline
from scoutmem_x.memory import (
    build_memory_snapshot,
    retrieve_best_node,
    retrieve_supporting_frames,
)
from scoutmem_x.perception import Detection
from scoutmem_x.tasks import run_passive_memory_search_episode

ROOT = Path(__file__).resolve().parents[1]


def test_memory_snapshot_merges_repeated_target_detections() -> None:
    from scoutmem_x.env import Observation

    observation_a = Observation(
        frame_id="frame-a",
        step_index=0,
        pose=(0.0, 0.0, 0.0),
        heading_radians=0.0,
        image_size=(128, 128),
        metadata={"scene_id": "merge_test"},
    )
    observation_b = Observation(
        frame_id="frame-b",
        step_index=1,
        pose=(1.0, 0.0, 0.0),
        heading_radians=0.0,
        image_size=(128, 128),
        metadata={"scene_id": "merge_test"},
    )
    detection_a = Detection(label="red mug", score=0.55, metadata={"region": "forward_cell"})
    detection_b = Detection(label="red mug", score=0.65, metadata={"region": "forward_cell"})

    snapshot_a = build_memory_snapshot(observation_a, [detection_a], "red mug")
    snapshot_b = build_memory_snapshot(observation_b, [detection_b], "red mug", snapshot_a)

    best_node = retrieve_best_node(snapshot_b, "red mug")
    assert best_node is not None
    assert len(snapshot_b.nodes) == 1
    assert best_node.confidence > 0.8
    assert retrieve_supporting_frames(snapshot_b, best_node.object_id) == ("frame-a", "frame-b")


def test_passive_memory_baseline_solves_unseen_hard_scene() -> None:
    config = AppConfig(
        phase="mvp2",
        subphase="2.5",
        mode="memory_eval",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
        scene_ids=("hall_unseen_hard",),
    )
    scene = next(scene for scene in load_default_scenes() if scene.scene_id == "hall_unseen_hard")

    result = run_passive_memory_search_episode(scene=scene, config=config)

    assert result.success is True
    assert result.final_memory.evidence_sufficiency_score > 0.8
    assert result.trace.final_action() is not None
    assert result.trace.final_action().action_type.value == "stop"


def test_passive_memory_eval_improves_unseen_success_rate() -> None:
    config = AppConfig(
        phase="mvp2",
        subphase="2.5",
        mode="baseline_compare",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
    )

    reactive_summary, memory_summary = compare_baselines(config)

    assert reactive_summary.success_rate == 0.5
    assert memory_summary.success_rate == 1.0
    assert memory_summary.average_steps == reactive_summary.average_steps


def test_cli_runs_memory_comparison(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    output_path = tmp_path / "mvp2_unseen_summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scoutmem_x.cli",
            "--config",
            "configs/mvp2_unseen.json",
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["reactive_summary"]["success_rate"] == 0.5
    assert payload["passive_memory_summary"]["success_rate"] == 1.0


def test_memory_only_eval_mode_runs() -> None:
    config = AppConfig(
        phase="mvp2",
        subphase="2.5",
        mode="memory_eval",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="seen",
    )
    summary = evaluate_passive_memory_baseline(config)

    assert summary.baseline == "passive_memory"
    assert summary.success_rate == 1.0
