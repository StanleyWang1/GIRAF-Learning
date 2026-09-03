import time

import numpy as np
import pytest

pytest.importorskip("cv2")

from dataset_visualization.backend.services.media import MediaService


class _SlowAdapter:
    def __init__(self):
        self.rgb_calls = []

    def frame_rgb(self, episode_index, stream_id, idx):
        _ = episode_index
        _ = stream_id
        self.rgb_calls.append(idx)
        time.sleep(0.03)
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def frame_depth(self, episode_index, stream_id, idx):
        _ = episode_index
        _ = stream_id
        _ = idx
        return np.zeros((16, 16), dtype=np.uint16)


class _LargeFrameAdapter(_SlowAdapter):
    def frame_rgb(self, episode_index, stream_id, idx):
        _ = episode_index
        _ = stream_id
        _ = idx
        return np.zeros((1200, 1600, 3), dtype=np.uint8)

    def frame_depth(self, episode_index, stream_id, idx):
        _ = episode_index
        _ = stream_id
        _ = idx
        return np.zeros((1200, 1600), dtype=np.uint16)


def test_get_rgb_does_not_block_on_prefetch_loop():
    adapter = _SlowAdapter()
    service = MediaService(adapter=adapter, prefetch=6)

    t0 = time.perf_counter()
    payload = service.get_rgb(episode_index=0, stream_id="cam0", idx=0)
    elapsed = time.perf_counter() - t0

    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert elapsed < 0.18

    service.shutdown()


def test_prefetch_dedupes_identical_tasks():
    adapter = _SlowAdapter()
    service = MediaService(adapter=adapter, prefetch=6)
    calls = {"count": 0}

    def _fake_prefetch_rgb(episode_index, stream_id, idx, profile="full"):
        _ = episode_index
        _ = stream_id
        _ = idx
        _ = profile
        calls["count"] += 1
        time.sleep(0.06)

    service._prefetch_rgb = _fake_prefetch_rgb  # type: ignore[method-assign]

    service._schedule_prefetch_rgb(episode_index=0, stream_id="cam0", idx=10)
    service._schedule_prefetch_rgb(episode_index=0, stream_id="cam0", idx=10)
    time.sleep(0.16)

    assert calls["count"] == 1
    service.shutdown()


def test_frame_profiles_reduce_preview_payload_size():
    adapter = _LargeFrameAdapter()
    service = MediaService(adapter=adapter, prefetch=0)

    full = service.get_rgb(episode_index=0, stream_id="cam0", idx=0, profile="full")
    preview = service.get_rgb(episode_index=0, stream_id="cam0", idx=0, profile="preview")

    assert isinstance(full, bytes)
    assert isinstance(preview, bytes)
    assert len(preview) < len(full)

    service.shutdown()
