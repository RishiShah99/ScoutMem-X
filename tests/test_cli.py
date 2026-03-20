from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cli_scaffold_output() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "scoutmem_x.cli", "--config", "configs/scaffold.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["config"]["phase"] == "mvp0"
    assert payload["config"]["subphase"] == "0.1"
    assert payload["message"] == "ScoutMem-X scaffold is ready for MVP-0."
