from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
pytest.importorskip("fastapi")
pytest.importorskip("starlette")
import zarr
from fastapi.testclient import TestClient

from dataset_visualization.backend.app import create_app
from dataset_visualization.backend.config import AppConfig


def _create_dataset(path: Path, episode_lengths=(2, 2, 2), full_zarr=False) -> np.ndarray:
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")

    total_steps = int(sum(episode_lengths))
    action = np.arange(total_steps * 3, dtype=np.float32).reshape(total_steps, 3)
    timestamps = np.linspace(0.0, float(total_steps - 1), total_steps, dtype=np.float64)
    episode_ends = np.cumsum(np.asarray(episode_lengths, dtype=np.int64), dtype=np.int64)

    data.array("action", action, chunks=(max(1, total_steps), 3), overwrite=True)
    data.array("timestamps", timestamps, chunks=(max(1, total_steps),), overwrite=True)
    meta.array("episode_ends", episode_ends, chunks=(max(1, len(episode_ends)),), overwrite=True)

    if full_zarr:
        rgb = np.zeros((total_steps, 1, 1, 3), dtype=np.uint8)
        data.array("camera_123_rgb", rgb, chunks=(1, 1, 1, 3), overwrite=True)

    return action


def _create_sidecar_dirs(videos_root: Path, episode_count: int) -> None:
    videos_root.mkdir(parents=True, exist_ok=True)
    for idx in range(episode_count):
        ep_dir = videos_root / f"ep_{idx}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / f"marker_{idx}.txt").write_text(f"episode-{idx}", encoding="utf-8")


def test_delete_episode_with_sidecar_compaction(tmp_path: Path):
    zpath = tmp_path / "raw.zarr"
    original_action = _create_dataset(zpath, episode_lengths=(2, 2, 2), full_zarr=False)
    videos_root = tmp_path / "raw_videos"
    _create_sidecar_dirs(videos_root, episode_count=3)

    config = AppConfig(
        input_path=zpath,
        videos_root=videos_root,
        episode=0,
        host="127.0.0.1",
        port=8080,
        history_frames=180,
        prefetch=0,
        depth_shape=None,
    )
    client = TestClient(create_app(config))

    resp = client.delete("/api/episode/1", params={"delete_videos": True})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["deleted_episode_index"] == 1
    assert payload["episode_count_before"] == 3
    assert payload["episode_count_after"] == 2
    assert payload["suggested_episode_index"] == 1
    assert payload["videos_applied"] is True
    assert payload["video_ops"]["deleted_dir"] == "ep_1"
    assert {"from": "ep_2", "to": "ep_1"} in payload["video_ops"]["renamed"]

    root = zarr.open(str(zpath), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    action_after = np.asarray(root["data"]["action"][:], dtype=np.float32)
    expected_action = np.vstack([original_action[:2], original_action[4:6]])

    assert episode_ends.tolist() == [2, 4]
    assert action_after.shape == (4, 3)
    np.testing.assert_allclose(action_after, expected_action)

    assert (videos_root / "ep_0").exists()
    assert (videos_root / "ep_1").exists()
    assert not (videos_root / "ep_2").exists()
    assert (videos_root / "ep_1" / "marker_2.txt").exists()


def test_delete_episode_sidecar_mismatch_fails_fast(tmp_path: Path):
    zpath = tmp_path / "raw_mismatch.zarr"
    _create_dataset(zpath, episode_lengths=(2, 2, 2), full_zarr=False)
    videos_root = tmp_path / "raw_mismatch_videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    (videos_root / "ep_0").mkdir(parents=True, exist_ok=True)
    (videos_root / "ep_2").mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        input_path=zpath,
        videos_root=videos_root,
        episode=0,
        host="127.0.0.1",
        port=8080,
        history_frames=180,
        prefetch=0,
        depth_shape=None,
    )
    client = TestClient(create_app(config))

    resp = client.delete("/api/episode/1", params={"delete_videos": True})
    assert resp.status_code == 409
    assert "sidecar episode folders are inconsistent" in resp.json()["detail"]

    root = zarr.open(str(zpath), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    assert episode_ends.tolist() == [2, 4, 6]
    assert (videos_root / "ep_0").exists()
    assert not (videos_root / "ep_1").exists()
    assert (videos_root / "ep_2").exists()


def test_delete_episode_full_zarr_without_sidecar(tmp_path: Path):
    zpath = tmp_path / "full.zarr"
    original_action = _create_dataset(zpath, episode_lengths=(2, 2), full_zarr=True)

    config = AppConfig(
        input_path=zpath,
        videos_root=None,
        episode=0,
        host="127.0.0.1",
        port=8080,
        history_frames=180,
        prefetch=0,
        depth_shape=None,
    )
    client = TestClient(create_app(config))

    resp = client.delete("/api/episode/0", params={"delete_videos": True})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["episode_count_before"] == 2
    assert payload["episode_count_after"] == 1
    assert payload["videos_applied"] is False
    assert payload["videos_root"] is None

    root = zarr.open(str(zpath), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    action_after = np.asarray(root["data"]["action"][:], dtype=np.float32)
    np.testing.assert_allclose(action_after, original_action[2:4])
    assert episode_ends.tolist() == [2]
