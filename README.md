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
- core typed schemas
- toy observe -> predict -> update -> act -> trace loop
- lightweight embodied search environment wrapper
- reactive frame-only search baseline
- passive-memory baseline with structured retrieval
- uncertainty-aware active evidence baseline
- seen/unseen evaluation configs and summaries
- initial test harness

See `AGENTS.md` for the phased build workflow and project conventions.
