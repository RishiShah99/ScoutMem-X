from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scoutmem_x.config import AppConfig
from scoutmem_x.tasks import run_toy_episode

ROOT = Path(__file__).resolve().parents[1]


def test_config_rejects_invalid_stop_threshold() -> None:
    with pytest.raises(ValueError, match="stop_threshold"):
        AppConfig(
            phase="mvp0",
            subphase="0.3",
            mode="toy_loop",
            max_steps=3,
            query="find the red mug",
            target_label="red mug",
            stop_threshold=1.1,
        )


def test_toy_episode_stops_after_confident_detection() -> None:
    config = AppConfig(
        phase="mvp0",
        subphase="0.3",
        mode="toy_loop",
        max_steps=4,
        query="find the red mug",
        target_label="red mug",
        stop_threshold=0.8,
    )

    result = run_toy_episode(config)

    assert result.trace.step_count == 3
    assert result.trace.final_action() is not None
    assert result.trace.final_action().action_type.value == "stop"
    assert result.trace.success is True
    assert result.final_memory.evidence_sufficiency_score >= 0.8
    assert len(result.final_memory.nodes) == 3


def test_cli_toy_loop_outputs_trace() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "scoutmem_x.cli", "--config", "configs/dev.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["config"]["subphase"] == "0.3"
    assert len(payload["trace"]["steps"]) == 3
    assert payload["trace"]["steps"][-1]["action"]["action_type"] == "stop"
    assert payload["final_memory"]["evidence_sufficiency_score"] >= 0.8
