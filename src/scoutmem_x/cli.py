from __future__ import annotations

import argparse
import json
from pathlib import Path

from scoutmem_x.config import load_config
from scoutmem_x.eval import (
    compare_active_baselines,
    compare_baselines,
    compare_stress_baselines,
    evaluate_active_evidence_baseline,
    evaluate_passive_memory_baseline,
    evaluate_reactive_baseline,
    evaluate_stress_baseline,
)
from scoutmem_x.serialization import to_jsonable
from scoutmem_x.tasks import run_toy_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ScoutMem-X scaffold CLI")
    parser.add_argument(
        "--config",
        default="configs/dev.json",
        help="Path to the JSON config file for the current scaffold run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON payload to disk.",
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
    elif config.mode == "baseline_eval":
        reactive_summary = evaluate_reactive_baseline(config)
        payload = {
            "status": "ok",
            "message": "ScoutMem-X reactive baseline evaluation executed.",
            "config": to_jsonable(config),
            "summary": to_jsonable(reactive_summary),
        }
    elif config.mode == "memory_eval":
        memory_summary = evaluate_passive_memory_baseline(config)
        payload = {
            "status": "ok",
            "message": "ScoutMem-X passive memory evaluation executed.",
            "config": to_jsonable(config),
            "summary": to_jsonable(memory_summary),
        }
    elif config.mode == "baseline_compare":
        reactive_summary, memory_summary = compare_baselines(config)
        payload = {
            "status": "ok",
            "message": "ScoutMem-X baseline comparison executed.",
            "config": to_jsonable(config),
            "reactive_summary": to_jsonable(reactive_summary),
            "passive_memory_summary": to_jsonable(memory_summary),
        }
    elif config.mode == "active_eval":
        active_summary = evaluate_active_evidence_baseline(config)
        payload = {
            "status": "ok",
            "message": "ScoutMem-X active evidence evaluation executed.",
            "config": to_jsonable(config),
            "summary": to_jsonable(active_summary),
        }
    elif config.mode == "active_compare":
        reactive_summary, memory_summary, active_summary = compare_active_baselines(config)
        payload = {
            "status": "ok",
            "message": "ScoutMem-X active baseline comparison executed.",
            "config": to_jsonable(config),
            "reactive_summary": to_jsonable(reactive_summary),
            "passive_memory_summary": to_jsonable(memory_summary),
            "active_evidence_summary": to_jsonable(active_summary),
        }
    elif config.mode == "stress_eval":
        stress_summary = evaluate_stress_baseline(config=config, baseline="active_evidence")
        payload = {
            "status": "ok",
            "message": "ScoutMem-X stress evaluation executed.",
            "config": to_jsonable(config),
            "summary": to_jsonable(stress_summary),
        }
    elif config.mode == "stress_compare":
        reactive_stress_summary, memory_stress_summary, active_stress_summary = (
            compare_stress_baselines(config)
        )
        payload = {
            "status": "ok",
            "message": "ScoutMem-X stress comparison executed.",
            "config": to_jsonable(config),
            "reactive_stress_summary": to_jsonable(reactive_stress_summary),
            "passive_memory_stress_summary": to_jsonable(memory_stress_summary),
            "active_evidence_stress_summary": to_jsonable(active_stress_summary),
        }
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{rendered}\n", encoding="utf-8")

    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
