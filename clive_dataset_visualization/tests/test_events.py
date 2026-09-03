import numpy as np

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.services.events import compute_events
from dataset_visualization.backend.types import DatasetSummary, EpisodeSchema


class DummyAdapter(BaseAdapter):
    format_name = "dummy"

    def __init__(self):
        super().__init__()
        self._keys = ["timestamps", "gripper_contact_L", "dagger", "action"]
        self._ts = np.arange(10, dtype=np.float64)
        self._contact = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=np.int8)
        self._dagger = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0], dtype=np.int8)
        self._action = np.zeros((10, 2), dtype=np.float32)
        self._action[6] = [10, 10]

    def dataset_summary(self):
        return DatasetSummary("dummy", "dummy", True, None, 1, 10)

    def episode_count(self):
        return 1

    def episode_bounds(self, episode_index):
        return (0, 10)

    def episode_schema(self, episode_index):
        return EpisodeSchema(
            episode_index=0,
            length=10,
            start=0,
            end=10,
            has_timestamps=True,
        )

    def episode_timestamps(self, episode_index, start, end, stride=1):
        return self._ts[start:end:stride]

    def signal_window(self, episode_index, key, start, end, stride=1):
        if key == "timestamps":
            return self._ts[start:end:stride]
        if key == "gripper_contact_L":
            return self._contact[start:end:stride]
        if key == "dagger":
            return self._dagger[start:end:stride]
        if key == "action":
            return self._action[start:end:stride]
        raise KeyError(key)

    def frame_rgb(self, episode_index, stream_id, idx):
        raise NotImplementedError

    def frame_depth(self, episode_index, stream_id, idx):
        raise NotImplementedError

    def list_stream_ids(self, episode_index):
        return []

    def has_depth(self, episode_index, stream_id):
        return False

    def graphable_keys(self, episode_index):
        return list(self._keys)

    def all_keys(self):
        return list(self._keys)


def test_compute_events_detects_contact_transitions():
    adapter = DummyAdapter()
    events = compute_events(adapter, 0)
    labels = [e["label"] for e in events]
    assert any("gripper_contact_L ON" in lbl for lbl in labels)
    assert any("gripper_contact_L OFF" in lbl for lbl in labels)


def test_compute_events_detects_dagger_transitions():
    adapter = DummyAdapter()
    events = compute_events(adapter, 0)
    labels = [e["label"] for e in events]
    assert any(lbl == "Dagger ON" for lbl in labels)
    assert any(lbl == "Dagger OFF" for lbl in labels)
