from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("zarr")
pytest.importorskip("fastapi")
pytest.importorskip("starlette")
import cv2
import zarr
from fastapi.testclient import TestClient

from dataset_visualization.backend.app import create_app
from dataset_visualization.backend.config import AppConfig


def _create_visual_dataset(path: Path) -> None:
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")

    frame_count = 3
    height = 800
    width = 1200
    y_grid, x_grid = np.indices((height, width))

    rgb_frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    for idx in range(frame_count):
        rgb_frames[idx, :, :, 0] = ((x_grid + idx * 11) % 255).astype(np.uint8)
        rgb_frames[idx, :, :, 1] = ((y_grid + idx * 17) % 255).astype(np.uint8)
        rgb_frames[idx, :, :, 2] = (((x_grid + y_grid) // 8 + idx * 23) % 255).astype(np.uint8)

    depth_frames = (y_grid.astype(np.uint16) * 4 + x_grid.astype(np.uint16) + 100).astype(np.uint16)
    depth_frames = np.repeat(depth_frames[np.newaxis, :, :], frame_count, axis=0)

    data.array("action", np.zeros((frame_count, 14), dtype=np.float32), chunks=(frame_count, 14), overwrite=True)
    data.array("timestamps", np.linspace(0.0, 0.2, frame_count), chunks=(frame_count,), dtype=np.float64, overwrite=True)
    data.array("camera_123_rgb", rgb_frames, chunks=(1, height, width, 3), overwrite=True)
    data.array("camera_123_depth", depth_frames, chunks=(1, height, width), overwrite=True)
    meta.array("episode_ends", np.array([frame_count], dtype=np.int64), chunks=(1,), overwrite=True)


def _decode_jpeg(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    assert frame is not None
    return frame


def test_frame_endpoint_supports_quality_profiles(tmp_path: Path):
    zpath = tmp_path / "frames.zarr"
    _create_visual_dataset(zpath)

    app = create_app(
        AppConfig(
            input_path=zpath,
            videos_root=None,
            episode=0,
            host="127.0.0.1",
            port=8080,
            history_frames=180,
            prefetch=0,
            depth_shape=None,
        )
    )
    client = TestClient(app)

    decoded = {}
    for profile in ("full", "scrub", "preview"):
        response = client.get(
            "/api/episode/0/frame",
            params={"camera": "123", "idx": 1, "modality": "rgb", "profile": profile},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/jpeg")
        decoded[profile] = _decode_jpeg(response.content)

    assert max(decoded["full"].shape[:2]) == 1200
    assert max(decoded["scrub"].shape[:2]) <= 960
    assert max(decoded["preview"].shape[:2]) <= 480
    assert max(decoded["full"].shape[:2]) > max(decoded["scrub"].shape[:2])
    assert max(decoded["scrub"].shape[:2]) > max(decoded["preview"].shape[:2])

    depth_response = client.get(
        "/api/episode/0/frame",
        params={"camera": "123", "idx": 1, "modality": "depth", "profile": "preview", "colormap": "turbo"},
    )
    assert depth_response.status_code == 200
    depth_frame = _decode_jpeg(depth_response.content)
    assert max(depth_frame.shape[:2]) <= 480
