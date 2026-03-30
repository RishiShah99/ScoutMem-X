# MVP-4 Review Note

## Completed Scope

- added a stress harness with deterministic perception perturbations
- added clean-vs-perturbed evaluation for reactive, passive-memory, and active-evidence baselines
- added a dedicated stress scene and config for robustness comparison

## What Worked

- the perturbation wrappers are modular and stay behind the perception interface
- the stress harness exposes a meaningful robustness gap between passive memory and active evidence
- earlier MVP behavior remains intact and fully tested

## Remaining Limitations

- perturbations are still lightweight and deterministic rather than visually realistic
- Habitat integration is still deferred intentionally so the behavior stack stays attributable
- robustness coverage is narrow and should expand before calling the project complete

## Ready Next

- start MVP-5 by wiring the real perception stack behind the existing adapter boundary, then rerun the clean and stress evaluations with noisy perception
