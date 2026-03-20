from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scoutmem_x.config import AppConfig
from scoutmem_x.eval import evaluate_reactive_baseline
from scoutmem_x.tasks import run_reactive_search_episode

ROOT = Path(__file__).resolve().parents[1]


def test_reactive_episode_succeeds_on_seen_scene() -> None:
    config = AppConfig(
        phase="mvp1",
        subphase="1.5",
        mode="baseline_eval",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="seen",
        scene_ids=("kitchen_seen_easy",),
    )
    from scoutmem_x.env import load_default_scenes

    scene = next(scene for scene in load_default_scenes() if scene.scene_id == "kitchen_seen_easy")
    result = run_reactive_search_episode(scene=scene, config=config)

    assert result.success is True
    assert result.trace.final_action() is not None
    assert result.trace.final_action().action_type.value == "stop"


def test_reactive_eval_reports_seen_summary() -> None:
    config = AppConfig(
        phase="mvp1",
        subphase="1.5",
        mode="baseline_eval",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="seen",
    )
    summary = evaluate_reactive_baseline(config)

    assert summary.total_episodes == 2
    assert summary.success_rate == 1.0
    assert summary.average_steps > 0.0


def test_reactive_eval_reports_unseen_summary() -> None:
    config = AppConfig(
        phase="mvp1",
        subphase="1.5",
        mode="baseline_eval",
        max_steps=5,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
        split="unseen",
    )
    summary = evaluate_reactive_baseline(config)

    assert summary.total_episodes == 2
    assert summary.success_rate == 0.5


def test_cli_runs_seen_eval_and_writes_summary(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    output_path = tmp_path / "mvp1_seen_summary.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scoutmem_x.cli",
            "--config",
            "configs/mvp1_seen.json",
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
    assert payload["summary"]["split"] == "seen"
    assert payload["summary"]["success_rate"] == 1.0
