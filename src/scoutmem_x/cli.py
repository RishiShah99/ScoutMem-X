from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from scoutmem_x.config import load_config


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

    payload = {
        "status": "ok",
        "message": "ScoutMem-X scaffold is ready for MVP-0.",
        "config": asdict(config),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
