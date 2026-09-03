from types import SimpleNamespace

import numpy as np

from dataset_visualization.backend.services.timing import compute_episode_timing


class _AdapterWithTimestamps:
    def __init__(self, timestamps):
        self._timestamps = np.asarray(timestamps, dtype=np.float64)

    def episode_schema(self, episode_index):
        _ = episode_index
        return SimpleNamespace(length=int(self._timestamps.shape[0]))

    def episode_timestamps(self, episode_index, start, end, stride=1):
        _ = episode_index
        return self._timestamps[start:end:stride]


def test_timing_infers_from_jittered_timestamps():
    dt = np.asarray([0.033, 0.034, 0.032, 0.035, 0.031, 0.033], dtype=np.float64)
    timestamps = np.concatenate([[0.0], np.cumsum(dt)])
    adapter = _AdapterWithTimestamps(timestamps)

    payload = compute_episode_timing(adapter, episode_index=0, fps_cap=30.0)

    assert payload["has_timestamps"] is True
    assert payload["frame_count"] == timestamps.shape[0]
    assert payload["median_dt_sec"] is not None
    assert payload["p90_dt_sec"] is not None
    assert payload["suggested_fps"] <= 30.0
    assert payload["suggested_fps"] >= 20.0


def test_timing_falls_back_when_timestamps_invalid():
    timestamps = np.asarray([0.0, 0.0, np.nan, 0.0, 0.0], dtype=np.float64)
    adapter = _AdapterWithTimestamps(timestamps)

    payload = compute_episode_timing(adapter, episode_index=0, fps_cap=30.0)

    assert payload["has_timestamps"] is False
    assert payload["median_dt_sec"] is None
    assert payload["p90_dt_sec"] is None
    assert payload["suggested_fps"] == 30.0
