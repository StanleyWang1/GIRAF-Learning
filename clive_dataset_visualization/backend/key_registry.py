from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set


GROUP_CORE = "Core"
GROUP_FORCES = "Forces"
GROUP_ACTION_VS_FOLLOWER = "Action vs Follower"
GROUP_LEADER_VS_FOLLOWER = "Leader vs Follower"
GROUP_ALL = "All"


def infer_group(key: str) -> str:
    if key in {"action", "action_follower", "timestamps", "joint_pos_L", "joint_pos_R", "gripper_pos_L", "gripper_pos_R"}:
        return GROUP_CORE

    if key.startswith("L_"):
        return GROUP_LEADER_VS_FOLLOWER

    if key in {"action", "action_follower"}:
        return GROUP_ACTION_VS_FOLLOWER

    if (
        "tau" in key
        or "wrench" in key
        or "contact" in key
        or key.startswith("robot0_wrench")
        or key.startswith("robot1_wrench")
    ):
        return GROUP_FORCES

    return GROUP_ALL


def build_key_groups(keys: Iterable[str], graphable_keys: Iterable[str]) -> Dict[str, List[str]]:
    keys_set: Set[str] = set(keys)
    graphable_set: Set[str] = set(graphable_keys)

    groups: Dict[str, List[str]] = {
        GROUP_CORE: [],
        GROUP_FORCES: [],
        GROUP_ACTION_VS_FOLLOWER: [],
        GROUP_LEADER_VS_FOLLOWER: [],
        GROUP_ALL: [],
    }

    for key in sorted(graphable_set):
        groups[GROUP_ALL].append(key)
        if key in {"action", "action_follower", "timestamps", "joint_pos_L", "joint_pos_R", "gripper_pos_L", "gripper_pos_R"}:
            groups[GROUP_CORE].append(key)
        if key in {"action", "action_follower"}:
            groups[GROUP_ACTION_VS_FOLLOWER].append(key)
        if key.startswith("L_"):
            groups[GROUP_LEADER_VS_FOLLOWER].append(key)
        if (
            "tau" in key
            or "wrench" in key
            or "contact" in key
            or key.startswith("robot0_wrench")
            or key.startswith("robot1_wrench")
        ):
            groups[GROUP_FORCES].append(key)

    # Keep predictable order and remove empties only where helpful for UI clarity.
    return groups


def parse_camera_key(key: str) -> tuple[str, str] | None:
    # camera_<id>_rgb, camera_<id>_depth, camera_<id>_rgb_cropped
    match = re.match(r"^camera_(.+?)_(rgb|depth)(?:_cropped)?$", key)
    if not match:
        return None

    stream_id = match.group(1)
    modality = match.group(2)

    if key.endswith("_rgb_cropped"):
        stream_id = f"{stream_id}_cropped"

    return stream_id, modality
