from __future__ import annotations

from scoutmem_x.env import GridSearchEnv, SearchSceneSpec, load_default_scenes
from scoutmem_x.policy import ActionType, AgentAction


def test_grid_search_env_reset_and_step() -> None:
    scene = SearchSceneSpec(
        scene_id="test_scene",
        split="seen",
        length=4,
        target_label="red mug",
        target_position=2,
    )
    env = GridSearchEnv(scene=scene)

    observation = env.reset()
    assert observation.metadata["agent_position"] == "0"

    transition = env.step(AgentAction(action_type=ActionType.MOVE_FORWARD, cost=1.0), step_index=0)
    assert transition.observation.metadata["agent_position"] == "1"
    assert transition.done is False


def test_default_scenes_include_seen_and_unseen_splits() -> None:
    scenes = load_default_scenes()
    splits = {scene.split for scene in scenes}
    assert "seen" in splits
    assert "unseen" in splits
