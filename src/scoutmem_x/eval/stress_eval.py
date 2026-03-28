from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.config import AppConfig
from scoutmem_x.eval.search_eval import EvalSummary, evaluate_baseline
from scoutmem_x.stress import StressPerceptionAdapter, get_perturbation_spec


@dataclass(frozen=True)
class StressSummary:
    baseline: str
    perturbation_name: str
    clean_summary: EvalSummary
    perturbed_summary: EvalSummary
    success_delta: float
    step_delta: float


def compare_stress_baselines(
    config: AppConfig,
) -> tuple[StressSummary, StressSummary, StressSummary]:
    return (
        evaluate_stress_baseline(config=config, baseline="reactive"),
        evaluate_stress_baseline(config=config, baseline="passive_memory"),
        evaluate_stress_baseline(config=config, baseline="active_evidence"),
    )


def evaluate_stress_baseline(config: AppConfig, baseline: str) -> StressSummary:
    if config.perturbation_name is None:
        raise ValueError("perturbation_name is required for stress evaluation")

    perturbation = get_perturbation_spec(config.perturbation_name)
    clean_summary = evaluate_baseline(config=config, baseline=baseline)
    stress_adapter = StressPerceptionAdapter(
        perturbation=perturbation,
        target_label=config.target_label,
    )
    perturbed_summary = evaluate_baseline(
        config=config,
        baseline=baseline,
        perception_adapter=stress_adapter,
    )
    return StressSummary(
        baseline=baseline,
        perturbation_name=perturbation.name,
        clean_summary=clean_summary,
        perturbed_summary=perturbed_summary,
        success_delta=perturbed_summary.success_rate - clean_summary.success_rate,
        step_delta=perturbed_summary.average_steps - clean_summary.average_steps,
    )
