from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scoutmem_x.config import AppConfig
from scoutmem_x.env import load_default_scenes
from scoutmem_x.eval import compare_active_baselines
from scoutmem_x.memory import MemorySnapshot
from scoutmem_x.perception import Detection
from scoutmem_x.policy import choose_active_evidence_action, estimate_uncertainty
from scoutmem_x.tasks import run_active_evidence_search_episode

ROOT = Path(__file__).resolve().parents[1]


def test_uncertainty_recommends_inspection_for_weak_visible_target() -> None:
    from scoutmem_x.env import Observation
    from scoutmem_x.memory import build_memory_snapshot

    observation_a = Observation(
        frame_id="frame-a",
        step_index=0,
        pose=(1.0, 0.0, 0.0),
        heading_radians=0.0,
        image_size=(128, 128),
        metadata={"scene_id": "attic_unseen_active"},
    )
    observation_b = Observation(
        frame_id="frame-b",
        step_index=1,
        pose=(2.0, 0.0, 0.0),
        heading_radians=0.0,
        image_size=(128, 128),
        metadata={"scene_id": "attic_unseen_active"},
    )
    detection_a = Detection(label="red mug", score=0.45, metadata={"region": "forward_cell"})
    detection_b = Detection(label="red mug", score=0.55, metadata={"region": "forward_cell"})

    snapshot_a = build_memory_snapshot(observation_a, [detection_a], "red mug")
    snapshot_b = build_memory_snapshot(observation_b, [detection_b], "red mug", snapshot_a)
    uncertainty = estimate_uncertainty(snapshot_b, [detection_b], "red mug", 0.8)

    assert uncertainty.confidence < 0.8
    assert uncertainty.inspect_recommended is True


def test_active_policy_chooses_inspect_before_stop() -> None:
    memory_snapshot = MemorySnapshot(
        evidence_sufficiency_score=0.74,
        revisitable_object_ids=("red mug-forward_cell",),
        target_object_id="red mug-forward_cell",
    )
    detections = [Detection(label="red mug", score=0.55)]

    action = choose_active_evidence_action(
        memory_snapshot=memory_snapshot,
        detections=detections,
        target_label="red mug",
        stop_threshold=0.8,
        max_steps=5,
        step_index=2,
    )

    assert action.action_type.value == "inspect"


def test_active_evidence_baseline_solves_active_challenge_scene() -> None:
    config = AppConfig(
        phase="mvp3",
        subphase="3.5",
        mode="active_eval",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
        scene_ids=("attic_unseen_active",),
    )
    scene = next(
        scene
        for scene in load_default_scenes()
        if scene.scene_id == "attic_unseen_active"
    )
    result = run_active_evidence_search_episode(scene=scene, config=config)

    actions = [step.action.action_type.value for step in result.trace.steps]
    assert result.success is True
    assert "inspect" in actions
    assert actions[-1] == "stop"
    assert result.steps_taken == 4


def test_active_comparison_improves_unseen_success_and_steps() -> None:
    config = AppConfig(
        phase="mvp3",
        subphase="3.5",
        mode="active_compare",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
        scene_ids=("garage_unseen_easy", "attic_unseen_active"),
    )
    reactive_summary, passive_summary, active_summary = compare_active_baselines(config)

    assert reactive_summary.success_rate == 0.5
    assert passive_summary.success_rate == 0.5
    assert active_summary.success_rate == 1.0
    assert active_summary.average_steps < passive_summary.average_steps


def test_cli_runs_active_comparison(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    output_path = tmp_path / "mvp3_unseen_summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scoutmem_x.cli",
            "--config",
            "configs/mvp3_unseen.json",
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
    assert payload["active_evidence_summary"]["success_rate"] == 1.0
    assert payload["passive_memory_summary"]["success_rate"] == 0.5
