from __future__ import annotations

from typing import Dict, List

import numpy as np

from dataset_visualization.backend.adapters.base import BaseAdapter


def _safe_timestamps(adapter: BaseAdapter, episode_index: int, length: int) -> np.ndarray:
    try:
        ts = adapter.episode_timestamps(episode_index, 0, length, 1)
        if len(ts) == length:
            return ts
    except Exception:
        pass
    return np.arange(length, dtype=np.float64)


def _append_event(events: List[Dict[str, object]], idx: int, ts: float, etype: str, label: str, details: Dict[str, object]) -> None:
    events.append(
        {
            "idx": int(idx),
            "time": float(ts),
            "type": etype,
            "label": label,
            "details": details,
        }
    )


def compute_events(adapter: BaseAdapter, episode_index: int, max_events: int = 2000) -> List[Dict[str, object]]:
    schema = adapter.episode_schema(episode_index)
    length = schema.length
    if length <= 0:
        return []

    timestamps = _safe_timestamps(adapter, episode_index, length)
    events: List[Dict[str, object]] = []
    keys = set(adapter.all_keys())

    # Contact transitions
    for key in ("gripper_contact_L", "gripper_contact_R"):
        if key not in keys:
            continue
        try:
            arr = adapter.signal_window(episode_index, key, 0, length, 1).reshape(-1)
        except Exception:
            continue
        if arr.size <= 1:
            continue
        arr_i = arr.astype(np.int8)
        transitions = np.where(np.diff(arr_i) != 0)[0] + 1
        for idx in transitions.tolist():
            state = int(arr_i[idx])
            label = f"{key} {'ON' if state else 'OFF'}"
            _append_event(events, idx, timestamps[idx], "contact", label, {"key": key, "state": state})
            if len(events) >= max_events:
                return sorted(events, key=lambda e: e["idx"])

    if "dagger" in keys:
        try:
            dagger_arr = adapter.signal_window(episode_index, "dagger", 0, length, 1).reshape(-1)
        except Exception:
            dagger_arr = np.asarray([], dtype=np.int8)

        if dagger_arr.size > 0:
            dagger_i = (dagger_arr.astype(np.float64) > 0.5).astype(np.int8)
            starts = np.where((dagger_i == 1) & np.concatenate(([True], dagger_i[:-1] == 0)))[0]
            ends = np.where((dagger_i == 1) & np.concatenate((dagger_i[1:] == 0, [True])))[0]

            for idx in starts.tolist():
                _append_event(events, idx, timestamps[idx], "dagger", "Dagger ON", {"key": "dagger", "state": 1})
                if len(events) >= max_events:
                    return sorted(events, key=lambda e: e["idx"])

            for idx in ends.tolist():
                _append_event(events, idx, timestamps[idx], "dagger", "Dagger OFF", {"key": "dagger", "state": 0})
                if len(events) >= max_events:
                    return sorted(events, key=lambda e: e["idx"])

    # Spike events on force/torque style keys, sampled for scalability.
    spike_keys = [
        k
        for k in keys
        if (
            "wrench" in k
            or "tau" in k
            or k.startswith("gripper_tau_external")
            or k.startswith("joints_tau_external")
        )
    ]
    spike_stride = max(1, length // 5000)
    sampled_ts = _safe_timestamps(adapter, episode_index, len(np.arange(0, length, spike_stride)))

    for key in sorted(spike_keys):
        try:
            arr = adapter.signal_window(episode_index, key, 0, length, spike_stride)
        except Exception:
            continue

        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            continue

        for c in range(arr.shape[1]):
            channel = arr[:, c].astype(np.float64)
            std = float(np.std(channel))
            if std < 1e-9:
                continue
            mean = float(np.mean(channel))
            z = np.abs((channel - mean) / std)
            spike_idx_local = np.where(z > 4.0)[0]
            for local_i in spike_idx_local.tolist():
                global_idx = local_i * spike_stride
                if global_idx >= length:
                    continue
                _append_event(
                    events,
                    global_idx,
                    timestamps[global_idx],
                    "spike",
                    f"Spike {key}[{c}]",
                    {"key": key, "channel": c, "zscore": float(z[local_i])},
                )
                if len(events) >= max_events:
                    return sorted(events, key=lambda e: e["idx"])

    # Action jump events
    for action_key in ("action", "action_follower"):
        if action_key not in keys:
            continue
        stride = max(1, length // 10000)
        try:
            action = adapter.signal_window(episode_index, action_key, 0, length, stride)
        except Exception:
            continue
        if action.ndim == 1:
            action = action[:, None]
        if action.shape[0] <= 2:
            continue

        diffs = np.linalg.norm(np.diff(action.astype(np.float64), axis=0), axis=1)
        mean = float(np.mean(diffs))
        std = float(np.std(diffs))
        threshold = mean + 4.0 * std
        if std < 1e-12:
            continue
        jump_local = np.where(diffs > threshold)[0] + 1
        for local_i in jump_local.tolist():
            idx = local_i * stride
            if idx >= length:
                continue
            _append_event(
                events,
                idx,
                timestamps[idx],
                "action_jump",
                f"Action jump: {action_key}",
                {"key": action_key, "magnitude": float(diffs[local_i - 1]), "threshold": threshold},
            )
            if len(events) >= max_events:
                return sorted(events, key=lambda e: e["idx"])

    # Missing modality flags
    stream_ids = adapter.list_stream_ids(episode_index)
    if not stream_ids:
        _append_event(events, 0, float(timestamps[0]), "warning", "No camera streams found", {})
    else:
        for sid in stream_ids:
            if not adapter.has_depth(episode_index, sid):
                _append_event(events, 0, float(timestamps[0]), "info", f"Depth missing for stream {sid}", {"stream_id": sid})

    events = sorted(events, key=lambda e: e["idx"])
    if len(events) > max_events:
        events = events[:max_events]
    return events
