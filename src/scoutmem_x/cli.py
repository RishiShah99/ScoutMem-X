from __future__ import annotations

import argparse
import json

from scoutmem_x.config import load_config
from scoutmem_x.serialization import to_jsonable
from scoutmem_x.tasks import run_toy_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScoutMem-X scaffold CLI")
    parser.add_argument(
        "--config",
        default="configs/dev.json",
        help="Path to the JSON config file for the current scaffold run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if config.mode == "scaffold":
        payload = {
            "status": "ok",
            "message": "ScoutMem-X scaffold is ready for MVP-0.",
            "config": to_jsonable(config),
        }
    elif config.mode == "toy_loop":
        result = run_toy_episode(config)
        payload = {
            "status": "ok",
            "message": "ScoutMem-X toy loop executed.",
            "config": to_jsonable(config),
            "trace": to_jsonable(result.trace),
            "final_memory": to_jsonable(result.final_memory),
        }
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
