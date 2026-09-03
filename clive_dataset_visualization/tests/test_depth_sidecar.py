from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
import zarr

from dataset_visualization.backend.adapters import raw_sidecar as raw_sidecar_mod
from dataset_visualization.backend.adapters.raw_sidecar import RawSidecarAdapter


def _create_raw_dataset(path: Path, n: int):
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    data.array("action", np.zeros((n, 14), dtype=np.float32), chunks=(n, 14), overwrite=True)
    data.array("timestamps", np.arange(n, dtype=np.float64), chunks=(n,), overwrite=True)
    meta.array("episode_ends", np.array([n], dtype=np.int64), chunks=(1,), overwrite=True)


def test_depth_shape_inference_common_case(tmp_path: Path):
    n = 4
    zpath = tmp_path / "raw_depth.zarr"
    _create_raw_dataset(zpath, n)

    videos_root = tmp_path / "raw_depth_videos"
    ep_dir = videos_root / "ep_0"
    ep_dir.mkdir(parents=True)

    h, w = 90, 160
    depth = (np.random.rand(n, h, w) * 1000).astype(np.uint16)
    depth_path = ep_dir / "camera_1_depth.bin"
    depth.tofile(depth_path)

    adapter = RawSidecarAdapter(zpath, videos_root=videos_root, depth_shape=None)
    schema = adapter.episode_schema(0)

    assert len(schema.cameras) == 1
    assert schema.cameras[0].stream_id == "1"
    assert schema.cameras[0].depth_shape == (h, w)

    frame = adapter.frame_depth(0, "1", 2)
    assert frame.shape == (h, w)


def test_reader_pool_reuses_capture_handles(tmp_path: Path, monkeypatch):
    n = 5
    zpath = tmp_path / "raw_rgb.zarr"
    _create_raw_dataset(zpath, n)

    videos_root = tmp_path / "raw_rgb_videos"
    ep_dir = videos_root / "ep_0"
    ep_dir.mkdir(parents=True)
    (ep_dir / "camera_1_rgb.mp4").touch()

    created_paths = []
    closed_paths = []

    class FakeReader:
        def __init__(self, video_path):
            self.video_path = video_path
            created_paths.append(str(video_path))

        def read(self, idx):
            _ = idx
            return np.zeros((4, 5, 3), dtype=np.uint8)

        def close(self):
            closed_paths.append(str(self.video_path))

    monkeypatch.setattr(raw_sidecar_mod, "_VideoReader", FakeReader)

    adapter = RawSidecarAdapter(zpath, videos_root=videos_root, depth_shape=None)
    frame0 = adapter.frame_rgb(0, "1", 0)
    frame1 = adapter.frame_rgb(0, "1", 1)
    frame4 = adapter.frame_rgb(0, "1", 4)

    assert frame0.shape == (4, 5, 3)
    assert frame1.shape == (4, 5, 3)
    assert frame4.shape == (4, 5, 3)
    assert len(created_paths) == 1

    adapter.close()
    assert len(closed_paths) == 1
