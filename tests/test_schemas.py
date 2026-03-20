from __future__ import annotations

import pytest

from scoutmem_x.env import Observation
from scoutmem_x.memory import MemoryNode, MemorySnapshot, Relation, VisibilityState
from scoutmem_x.perception import Detection
from scoutmem_x.policy import ActionType, AgentAction
from scoutmem_x.tasks import EpisodeStepRecord, EpisodeTrace


def test_observation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="step_index"):
        Observation(
            frame_id="frame-0",
            step_index=-1,
            pose=(0.0, 0.0, 0.0),
            heading_radians=0.0,
            image_size=(64, 64),
        )

    with pytest.raises(ValueError, match="image_size"):
        Observation(
            frame_id="frame-1",
            step_index=0,
            pose=(0.0, 0.0, 0.0),
            heading_radians=0.0,
            image_size=(0, 64),
        )


def test_memory_node_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="query_match_score"):
        MemoryNode(
            object_id="obj-1",
            category="mug",
            query_match_score=1.2,
            confidence=0.4,
            last_seen_step=0,
            visibility_state=VisibilityState.VISIBLE,
        )

    with pytest.raises(ValueError, match="confidence"):
        MemoryNode(
            object_id="obj-1",
            category="mug",
            query_match_score=0.9,
            confidence=-0.1,
            last_seen_step=0,
            visibility_state=VisibilityState.VISIBLE,
        )


def test_memory_snapshot_rejects_invalid_sufficiency() -> None:
    with pytest.raises(ValueError, match="evidence_sufficiency_score"):
        MemorySnapshot(evidence_sufficiency_score=1.5)


def test_agent_action_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="cost"):
        AgentAction(action_type=ActionType.EXPLORE, cost=-1.0)


def test_episode_trace_links_core_schemas() -> None:
    observation = Observation(
        frame_id="frame-2",
        step_index=2,
        pose=(1.0, 2.0, 0.0),
        heading_radians=1.57,
        image_size=(128, 128),
        rgb_path="outputs/frame-2.png",
    )
    detection = Detection(label="mug", score=0.8, region=(10, 12, 40, 44))
    action = AgentAction(action_type=ActionType.INSPECT, target_id="obj-1", cost=0.2)
    memory = MemorySnapshot(
        nodes=(
            MemoryNode(
                object_id="obj-1",
                category="mug",
                query_match_score=0.95,
                confidence=0.8,
                last_seen_step=2,
                visibility_state=VisibilityState.VISIBLE,
                room_or_region_estimate="kitchen",
                supporting_frames=("frame-2",),
                relations=(Relation(relation_type="near", target="table"),),
            ),
        ),
        unexplored_regions=("kitchen_corner",),
        revisitable_object_ids=("obj-1",),
        evidence_sufficiency_score=0.6,
    )
    step = EpisodeStepRecord(
        observation=observation,
        detections=(detection,),
        action=action,
        memory_snapshot=memory,
        notes=("first positive evidence",),
    )
    trace = EpisodeTrace(
        episode_id="episode-1",
        query="find the mug",
        steps=(step,),
        success=None,
    )

    assert trace.step_count == 1
    assert trace.final_action() == action
    assert trace.steps[0].memory_snapshot.nodes[0].category == "mug"
