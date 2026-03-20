from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    phase: str
    subphase: str
    mode: str
    max_steps: int
    query: str


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(
        phase=str(raw["phase"]),
        subphase=str(raw["subphase"]),
        mode=str(raw["mode"]),
        max_steps=int(raw["max_steps"]),
        query=str(raw["query"]),
    )
