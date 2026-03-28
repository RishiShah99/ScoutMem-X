from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scoutmem_x.config import AppConfig
from scoutmem_x.env import GridSearchEnv, load_default_scenes
from scoutmem_x.eval import compare_stress_baselines, evaluate_stress_baseline
from scoutmem_x.perception import OraclePerceptionAdapter
from scoutmem_x.stress import StressPerceptionAdapter, get_perturbation_spec

ROOT = Path(__file__).resolve().parents[1]


def test_target_dropout_perturbation_removes_target_detection() -> None:
    scene = next(
        scene
        for scene in load_default_scenes()
        if scene.scene_id == "basement_unseen_stress"
    )
    env = GridSearchEnv(scene=scene)
    env.reset()
    env.agent_position = 2
    observation = env._build_observation(step_index=2)  # noqa: SLF001

    adapter = StressPerceptionAdapter(
        perturbation=get_perturbation_spec("drop_first_target_glimpse"),
        target_label="red mug",
        base_adapter=OraclePerceptionAdapter(),
    )

    detections = adapter.predict(observation=observation, query="find the red mug")
    assert detections == []


def test_false_positive_perturbation_injects_target_detection() -> None:
    scene = next(scene for scene in load_default_scenes() if scene.scene_id == "garage_unseen_easy")
    env = GridSearchEnv(scene=scene)
    observation = env.reset()

    adapter = StressPerceptionAdapter(
        perturbation=get_perturbation_spec("inject_false_target"),
        target_label="red mug",
        base_adapter=OraclePerceptionAdapter(),
    )

    detections = adapter.predict(observation=observation, query="find the red mug")
    assert any(detection.label == "red mug" for detection in detections)


def test_stress_comparison_preserves_active_success() -> None:
    config = AppConfig(
        phase="mvp4",
        subphase="4.5",
        mode="stress_compare",
        max_steps=6,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
        scene_ids=("garage_unseen_easy", "basement_unseen_stress"),
        perturbation_name="drop_first_target_glimpse",
    )

    reactive_summary, passive_summary, active_summary = compare_stress_baselines(config)

    assert reactive_summary.clean_summary.success_rate == 0.5
    assert reactive_summary.perturbed_summary.success_rate == 0.5
    assert passive_summary.clean_summary.success_rate == 1.0
    assert passive_summary.perturbed_summary.success_rate == 0.5
    assert active_summary.clean_summary.success_rate == 1.0
    assert active_summary.perturbed_summary.success_rate == 1.0
    assert active_summary.success_delta == 0.0
    assert passive_summary.success_delta == -0.5


def test_active_stress_eval_mode_runs() -> None:
    config = AppConfig(
        phase="mvp4",
        subphase="4.5",
        mode="stress_eval",
        max_steps=6,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
        scene_ids=("garage_unseen_easy", "basement_unseen_stress"),
        perturbation_name="drop_first_target_glimpse",
    )

    summary = evaluate_stress_baseline(config=config, baseline="active_evidence")
    assert summary.baseline == "active_evidence"
    assert summary.perturbed_summary.success_rate == 1.0


def test_cli_runs_stress_comparison(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    output_path = tmp_path / "mvp4_stress_summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scoutmem_x.cli",
            "--config",
            "configs/mvp4_stress.json",
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
    assert payload["active_evidence_stress_summary"]["perturbed_summary"]["success_rate"] == 1.0
    assert payload["passive_memory_stress_summary"]["perturbed_summary"]["success_rate"] == 0.5
