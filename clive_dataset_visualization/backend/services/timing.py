from __future__ import annotations

from typing import Dict

import numpy as np

from dataset_visualization.backend.adapters.base import BaseAdapter


DEFAULT_FPS = 30.0
MIN_DT = 1e-6


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return float(value)


def compute_episode_timing(adapter: BaseAdapter, episode_index: int, fps_cap: float = DEFAULT_FPS) -> Dict[str, float | int | bool | None]:
    schema = adapter.episode_schema(episode_index)
    frame_count = int(schema.length)

    if frame_count <= 1:
        return {
            "has_timestamps": False,
            "median_dt_sec": None,
            "p90_dt_sec": None,
            "suggested_fps": float(fps_cap),
            "frame_count": frame_count,
        }

    try:
        timestamps = np.asarray(adapter.episode_timestamps(episode_index, 0, frame_count, 1), dtype=np.float64)
    except Exception:
        timestamps = np.asarray([], dtype=np.float64)

    if timestamps.shape[0] != frame_count:
        return {
            "has_timestamps": False,
            "median_dt_sec": None,
            "p90_dt_sec": None,
            "suggested_fps": float(fps_cap),
            "frame_count": frame_count,
        }

    diffs = np.diff(timestamps)
    valid = np.isfinite(diffs) & (diffs > MIN_DT)
    valid_diffs = diffs[valid]

    if valid_diffs.shape[0] < max(3, int(0.5 * diffs.shape[0])):
        # If less than half the steps are usable, timestamps are not reliable enough.
        return {
            "has_timestamps": False,
            "median_dt_sec": None,
            "p90_dt_sec": None,
            "suggested_fps": float(fps_cap),
            "frame_count": frame_count,
        }

    median_dt = float(np.median(valid_diffs))
    p90_dt = float(np.percentile(valid_diffs, 90.0))

    if median_dt <= MIN_DT:
        return {
            "has_timestamps": False,
            "median_dt_sec": None,
            "p90_dt_sec": None,
            "suggested_fps": float(fps_cap),
            "frame_count": frame_count,
        }

    inferred_fps = 1.0 / median_dt
    suggested_fps = float(max(1.0, min(float(fps_cap), inferred_fps)))

    return {
        "has_timestamps": True,
        "median_dt_sec": _safe_float(median_dt),
        "p90_dt_sec": _safe_float(p90_dt),
        "suggested_fps": suggested_fps,
        "frame_count": frame_count,
    }
