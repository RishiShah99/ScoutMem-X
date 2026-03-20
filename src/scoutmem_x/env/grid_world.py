from __future__ import annotations

from dataclasses import dataclass

from scoutmem_x.env.observation import Observation
from scoutmem_x.env.scene import EnvironmentStep, SearchSceneSpec
from scoutmem_x.policy.actions import ActionType, AgentAction

DEFAULT_SCENES: tuple[SearchSceneSpec, ...] = (
    SearchSceneSpec(
        scene_id="kitchen_seen_easy",
        split="seen",
        length=5,
        target_label="red mug",
        target_position=2,
        distractors={1: "blue bowl"},
    ),
    SearchSceneSpec(
        scene_id="office_seen_medium",
        split="seen",
        length=6,
        target_label="red mug",
        target_position=3,
        distractors={2: "red book"},
    ),
    SearchSceneSpec(
        scene_id="garage_unseen_easy",
        split="unseen",
        length=5,
        target_label="red mug",
        target_position=2,
        distractors={0: "paint can"},
    ),
    SearchSceneSpec(
        scene_id="hall_unseen_hard",
        split="unseen",
        length=7,
        target_label="red mug",
        target_position=6,
        distractors={3: "red bottle"},
    ),
)


def load_default_scenes() -> tuple[SearchSceneSpec, ...]:
    return DEFAULT_SCENES


@dataclass
class GridSearchEnv:
    scene: SearchSceneSpec
    agent_position: int = 0
    heading_radians: float = 0.0

    def reset(self) -> Observation:
        self.agent_position = 0
        self.heading_radians = 0.0
        return self._build_observation(step_index=0)

    def step(self, action: AgentAction, step_index: int) -> EnvironmentStep:
        if action.action_type == ActionType.MOVE_FORWARD:
            self.agent_position = min(self.agent_position + 1, self.scene.length - 1)
        elif action.action_type == ActionType.ROTATE_LEFT:
            self.heading_radians -= 0.25
        elif action.action_type == ActionType.ROTATE_RIGHT:
            self.heading_radians += 0.25

        found_target = (
            action.action_type == ActionType.STOP
            and self.agent_position == self.scene.target_position
        )
        observation = self._build_observation(step_index=step_index + 1)
        done = action.action_type == ActionType.STOP or found_target
        reward = 1.0 if found_target else 0.0
        return EnvironmentStep(
            observation=observation,
            reward=reward,
            done=done,
            found_target=found_target,
        )

    def _build_observation(self, step_index: int) -> Observation:
        visible_label, visible_score = self._visible_object()
        metadata = {
            "scene_id": self.scene.scene_id,
            "split": self.scene.split,
            "agent_position": str(self.agent_position),
            "visible_label": visible_label,
            "visible_score": f"{visible_score:.2f}",
            "visible_region": "forward_cell",
        }
        return Observation(
            frame_id=f"{self.scene.scene_id}-frame-{step_index}",
            step_index=step_index,
            pose=(float(self.agent_position), 0.0, 0.0),
            heading_radians=self.heading_radians,
            image_size=(128, 128),
            metadata=metadata,
        )

    def _visible_object(self) -> tuple[str, float]:
        if self.agent_position == self.scene.target_position:
            return self.scene.target_label, 0.95
        distractor = self.scene.distractors.get(self.agent_position)
        if distractor is not None:
            return distractor, 0.55
        return "", 0.0
