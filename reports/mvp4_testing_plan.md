# MVP-4 Testing Plan

Scope: robustness stress harness with perturbation wrappers and clean-vs-perturbed baseline comparisons.

## Tier A Dev Slice

- run `configs/mvp4_stress.json`
- save one stress comparison artifact to `outputs/`

## Unit Coverage

- target-dropout perturbation removes the expected target detection
- false-positive perturbation injects a synthetic target detection

## Integration Coverage

- stress comparison evaluates reactive, passive-memory, and active-evidence baselines
- CLI stress mode emits clean and perturbed summaries for all baselines

## Regression Coverage

- MVP-0 through MVP-3 test suites remain green
- MVP-2 memory improvement and MVP-3 active evidence improvement remain intact

## Exit Criteria

- passive-memory clean success exceeds perturbed passive-memory success on the stress slice
- active-evidence perturbed success remains at `1.0` on the stress slice
- `python -m pytest` passes
- `python -m ruff check .` passes
- `python -m mypy src` passes
