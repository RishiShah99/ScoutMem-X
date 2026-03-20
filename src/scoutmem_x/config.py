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
    target_label: str
    stop_threshold: float

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError("stop_threshold must be between 0.0 and 1.0")


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(
        phase=str(raw["phase"]),
        subphase=str(raw["subphase"]),
        mode=str(raw["mode"]),
        max_steps=int(raw["max_steps"]),
        query=str(raw["query"]),
        target_label=str(raw.get("target_label", raw["query"])),
        stop_threshold=float(raw.get("stop_threshold", 0.8)),
    )
