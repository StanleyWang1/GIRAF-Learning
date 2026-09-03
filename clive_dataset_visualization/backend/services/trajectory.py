from __future__ import annotations

from typing import Dict

from dataset_visualization.backend.adapters.base import BaseAdapter


def build_trajectory_payload(adapter: BaseAdapter, episode_index: int, idx: int, window: int) -> Dict[str, object]:
    schema = adapter.episode_schema(episode_index)
    length = schema.length
    if length <= 0:
        return {"available": False, "reason": "Empty episode"}

    idx = max(0, min(length - 1, int(idx)))
    window = max(1, int(window))
    start = max(0, idx - window + 1)
    end = idx + 1

    keys = set(adapter.all_keys())
    payload: Dict[str, object] = {
        "available": False,
        "idx": idx,
        "start": start,
        "end": end,
        "robots": {},
    }

    robot_specs = [
        ("robot0", "robot0_eef_pos", "robot0_eef_rpy"),
        ("robot1", "robot1_eef_pos", "robot1_eef_rpy"),
    ]

    for robot_name, pos_key, rpy_key in robot_specs:
        if pos_key not in keys:
            continue
        try:
            pos = adapter.signal_window(episode_index, pos_key, start, end, 1)
            rpy = adapter.signal_window(episode_index, rpy_key, start, end, 1) if rpy_key in keys else None
        except Exception:
            continue

        if pos.ndim != 2 or pos.shape[1] < 3:
            continue

        payload["robots"][robot_name] = {
            "history_pos": pos[:, :3].astype(float).tolist(),
            "current_pos": pos[-1, :3].astype(float).tolist(),
            "history_rpy": rpy[:, :3].astype(float).tolist() if rpy is not None and rpy.ndim == 2 and rpy.shape[1] >= 3 else None,
            "current_rpy": rpy[-1, :3].astype(float).tolist() if rpy is not None and rpy.ndim == 2 and rpy.shape[1] >= 3 else None,
        }

    if payload["robots"]:
        payload["available"] = True
    else:
        payload["reason"] = "No robot*_eef_pos keys available"

    return payload
