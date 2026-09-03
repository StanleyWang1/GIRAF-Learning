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


def _create_dataset(path: Path):
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    data.array("action", np.zeros((5, 14), dtype=np.float32), chunks=(5, 14), overwrite=True)
    data.array("timestamps", np.linspace(0.0, 0.4, 5), chunks=(5,), dtype=np.float64, overwrite=True)
    data.array("joint_pos_L", np.zeros((5, 6), dtype=np.float32), chunks=(5, 6), overwrite=True)
    data.array("gripper_pos_L", np.zeros((5,), dtype=np.float32), chunks=(5,), overwrite=True)
    meta.array("episode_ends", np.array([5], dtype=np.int64), chunks=(1,), overwrite=True)


def test_api_smoke(tmp_path: Path):
    zpath = tmp_path / "smoke.zarr"
    _create_dataset(zpath)

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
    app = create_app(config)
    client = TestClient(app)

    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/dataset/summary")
    assert r.status_code == 200
    assert r.json()["supported"] is True

    r = client.get("/api/episodes")
    assert r.status_code == 200
    payload = r.json()
    assert payload["episode_count"] == 1

    r = client.get("/api/episode/0/schema")
    assert r.status_code == 200
    assert r.json()["length"] == 5

    r = client.get("/api/episode/0/timing")
    assert r.status_code == 200
    timing = r.json()
    assert set(["has_timestamps", "median_dt_sec", "p90_dt_sec", "suggested_fps", "frame_count"]).issubset(set(timing.keys()))
    assert timing["frame_count"] == 5

    r = client.get("/api/episode/0/signals", params={"keys": "action", "start": 0, "end": 5, "stride": 1})
    assert r.status_code == 200
    signals = r.json()
    assert signals["episode_index"] == 0
    assert "action" in signals["series"]

    r = client.get("/api/episode/0/events")
    assert r.status_code == 200
    events = r.json()
    assert events["episode_index"] == 0
    assert isinstance(events["events"], list)

    r = client.get("/api/episode/0/trajectory3d", params={"idx": 4, "window": 5})
    assert r.status_code == 200
    trajectory = r.json()
    assert "available" in trajectory
