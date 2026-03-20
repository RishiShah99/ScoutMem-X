# ScoutMem-X

Phased research engineering project for embodied search under partial observability.

## Quick Start

```bash
conda create -n sceneforge python=3.10.19 -y
conda activate sceneforge
python -m pip install -e .[dev]
python -m scoutmem_x.cli --config configs/dev.json
python -m pytest
```

## Current Scope

- minimal repo scaffold
- package + CLI entry point
- config loading
- perception adapter contract
- initial test harness

See `AGENTS.md` for the phased build workflow and project conventions.
