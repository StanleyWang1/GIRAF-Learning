from __future__ import annotations

from typing import Dict, List

import numpy as np

from dataset_visualization.backend.adapters.base import BaseAdapter


def _to_numeric_list(values: np.ndarray) -> List[float]:
    if values.dtype.kind == "b":
        return values.astype(np.int8).astype(float).tolist()
    return values.astype(np.float64).tolist()


def build_signal_payload(
    adapter: BaseAdapter,
    episode_index: int,
    keys: List[str],
    start: int,
    end: int,
    stride: int,
) -> Dict[str, object]:
    schema = adapter.episode_schema(episode_index)
    length = schema.length

    start = max(0, min(int(start), length))
    end = max(start, min(int(end), length))
    stride = max(1, int(stride))

    indices = np.arange(start, end, stride, dtype=np.int64)
    timestamps = adapter.episode_timestamps(episode_index, start, end, stride)
    if len(timestamps) != len(indices):
        # Fallback if timestamps are unavailable or malformed.
        timestamps = indices.astype(np.float64)

    series: Dict[str, List[Dict[str, object]]] = {}
    skipped: List[str] = []

    for key in keys:
        try:
            arr = adapter.signal_window(episode_index, key, start, end, stride)
        except Exception:
            skipped.append(key)
            continue

        if arr.ndim == 1:
            series[key] = [{"name": key, "values": _to_numeric_list(arr)}]
            continue

        if arr.ndim == 2:
            key_channels: List[Dict[str, object]] = []
            for i in range(arr.shape[1]):
                key_channels.append({"name": f"{key}[{i}]", "values": _to_numeric_list(arr[:, i])})
            series[key] = key_channels
            continue

        skipped.append(key)

    return {
        "episode_index": episode_index,
        "start": start,
        "end": end,
        "stride": stride,
        "indices": indices.tolist(),
        "timestamps": timestamps.astype(np.float64).tolist(),
        "series": series,
        "skipped": skipped,
    }
