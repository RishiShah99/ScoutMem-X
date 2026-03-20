from __future__ import annotations

from scoutmem_x.memory.schema import MemoryNode, MemorySnapshot


def retrieve_best_node(memory_snapshot: MemorySnapshot, target_label: str) -> MemoryNode | None:
    candidates = tuple(node for node in memory_snapshot.nodes if node.category == target_label)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda node: (node.confidence, node.query_match_score, node.last_seen_step),
    )


def retrieve_supporting_frames(memory_snapshot: MemorySnapshot, object_id: str) -> tuple[str, ...]:
    for node in memory_snapshot.nodes:
        if node.object_id == object_id:
            return node.supporting_frames
    return ()
