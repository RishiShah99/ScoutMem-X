# MVP-4 Eval Summary

- `perturbation`: `drop_first_target_glimpse`
- `reactive clean success_rate`: `0.50`
- `reactive perturbed success_rate`: `0.50`
- `passive_memory clean success_rate`: `1.00`
- `passive_memory perturbed success_rate`: `0.50`
- `active_evidence clean success_rate`: `1.00`
- `active_evidence perturbed success_rate`: `1.00`

## Notes

- the stress slice drops the first weak target glimpse in the dedicated stress scene
- passive memory degrades because persistence alone is not enough when the first glimpse is removed
- active evidence remains successful by inspecting the later weak target evidence until confidence is sufficient
