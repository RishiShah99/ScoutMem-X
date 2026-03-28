from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from scoutmem_x.perception import Detection, OraclePerceptionAdapter, PerceptionAdapter


class PerturbationKind(str, Enum):
    TARGET_DROPOUT = "target_dropout"
    FALSE_POSITIVE = "false_positive"
    SCORE_DECAY = "score_decay"


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    kind: PerturbationKind
    scene_positions: Mapping[str, tuple[int, ...]]
    score_scale: float = 1.0
    injected_label: str = ""
    injected_score: float = 0.0


DEFAULT_PERTURBATIONS: tuple[PerturbationSpec, ...] = (
    PerturbationSpec(
        name="drop_first_target_glimpse",
        kind=PerturbationKind.TARGET_DROPOUT,
        scene_positions={"basement_unseen_stress": (2,)},
    ),
    PerturbationSpec(
        name="inject_false_target",
        kind=PerturbationKind.FALSE_POSITIVE,
        scene_positions={"garage_unseen_easy": (0,)},
        injected_label="red mug",
        injected_score=0.82,
    ),
    PerturbationSpec(
        name="weaken_target_scores",
        kind=PerturbationKind.SCORE_DECAY,
        scene_positions={"hall_unseen_hard": (3, 4), "attic_unseen_active": (1, 2)},
        score_scale=0.75,
    ),
)


def get_perturbation_spec(name: str) -> PerturbationSpec:
    for spec in DEFAULT_PERTURBATIONS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown perturbation: {name}")


class StressPerceptionAdapter:
    def __init__(
        self,
        perturbation: PerturbationSpec,
        target_label: str,
        base_adapter: PerceptionAdapter | None = None,
    ) -> None:
        self._perturbation = perturbation
        self._target_label = target_label
        self._base_adapter = base_adapter or OraclePerceptionAdapter()

    def predict(self, observation: object, query: str) -> list[Detection]:
        detections = self._base_adapter.predict(observation=observation, query=query)
        metadata = getattr(observation, "metadata", {})
        scene_id = str(metadata.get("scene_id", ""))
        agent_position = int(metadata.get("agent_position", "0"))
        affected_positions = self._perturbation.scene_positions.get(scene_id, ())

        if agent_position not in affected_positions:
            return detections

        if self._perturbation.kind == PerturbationKind.TARGET_DROPOUT:
            return [detection for detection in detections if detection.label != self._target_label]

        if self._perturbation.kind == PerturbationKind.SCORE_DECAY:
            return [self._decay_detection(detection) for detection in detections]

        if self._perturbation.kind == PerturbationKind.FALSE_POSITIVE:
            return self._inject_false_positive(detections)

        return detections

    def _decay_detection(self, detection: Detection) -> Detection:
        if detection.label != self._target_label:
            return detection
        return Detection(
            label=detection.label,
            score=max(detection.score * self._perturbation.score_scale, 0.0),
            region=detection.region,
            embedding=detection.embedding,
            mask=detection.mask,
            metadata=detection.metadata,
        )

    def _inject_false_positive(self, detections: list[Detection]) -> list[Detection]:
        if any(detection.label == self._target_label for detection in detections):
            return detections
        false_positive = Detection(
            label=self._perturbation.injected_label or self._target_label,
            score=self._perturbation.injected_score,
            region=(0, 0, 12, 12),
            metadata={"source": "stress_false_positive", "region": "forward_cell"},
        )
        return [*detections, false_positive]
