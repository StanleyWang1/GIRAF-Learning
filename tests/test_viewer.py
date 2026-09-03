from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest
import zarr

from giraf.viewer.dataset import DatasetFormatError, GirafDataset
from giraf.viewer.server import create_server


def _write_dataset(path: Path) -> Path:
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        {
            "schema_version": "giraf-replay-v1",
            "aligned_hz": 10.0,
            "action_fields": [f"action_{index}" for index in range(7)],
            "state_fields": [f"state_{index}" for index in range(15)],
            "joint_fields": [f"joint_{index}" for index in range(6)],
            "clean": {
                "segments": [
                    {"source_episode": 14, "start": 100, "stop": 103},
                    {"source_episode": 15, "start": 103, "stop": 105},
                ]
            },
        }
    )
    data = root.require_group("data")
    meta = root.require_group("meta")

    camera = np.zeros((5, 8, 10, 3), dtype=np.uint8)
    camera[0, :, :, 0] = 255
    camera[1, :, :, 1] = 255
    camera[2, :, :, 2] = 255
    data.array("camera_rgb", camera, chunks=(2, 8, 10, 3))
    data.array("timestamp", np.arange(5, dtype=np.float64) / 10 + 10.0, chunks=(2,))
    data.array("timestamp_ns", np.arange(5, dtype=np.int64) * 100_000_000, chunks=(2,))

    action = np.zeros((5, 7), dtype=np.float32)
    action[1, 0] = 0.25
    action[2, 1] = -0.5
    data.array("action", action, chunks=(2, 7))
    data.array("state", np.arange(75, dtype=np.float32).reshape(5, 15), chunks=(2, 15))
    data.array("alignment_valid", np.array([1, 0, 1, 1, 1], dtype=np.uint8), chunks=(2,))
    data.array("tracking", np.ones(5, dtype=np.uint8), chunks=(2,))
    data.array("clutch", np.ones(5, dtype=np.uint8), chunks=(2,))
    data.array("motor_command_accepted", np.ones(5, dtype=np.uint8), chunks=(2,))
    data.array("grasp_label", np.array([0, 1, 1, 0, 0], dtype=np.uint8), chunks=(2,))
    data.array("camera_sequence_num", np.array([10, 11, 12, 20, 21], dtype=np.int64), chunks=(2,))
    data.array("control_age_ns", np.full(5, 2_000_000, dtype=np.int64), chunks=(2,))
    data.array("motor_age_ns", np.full(5, 3_000_000, dtype=np.int64), chunks=(2,))

    meta.array("episode_ends", np.array([3, 5], dtype=np.int64), chunks=(2,))
    meta.array("episode_valid_steps", np.array([2, 2], dtype=np.int64), chunks=(2,))
    meta.array("episode_invalid_steps", np.array([1, 0], dtype=np.int64), chunks=(2,))
    return path


def test_giraf_dataset_exposes_episodes_labels_metrics_and_events(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path / "sample.zarr")
    dataset = GirafDataset(path)

    assert dataset.episode_count == 2
    assert dataset.summary(1)["requested_episode"] == 1
    assert dataset.episode_bounds(1) == (3, 5)
    assert dataset.timestamps(0).tolist() == pytest.approx([0.0, 0.1, 0.2])

    metrics = dataset.episode_metrics(0)
    assert metrics["source_episode"] == 14
    assert metrics["valid_steps"] == 2
    assert metrics["valid_ratio"] == pytest.approx(2 / 3)
    assert metrics["metadata_counts_match"] is True

    schema = dataset.schema(0)
    action_info = next(item for item in schema["keys"] if item["key"] == "action")
    assert action_info["channels"] == [f"action_{index}" for index in range(7)]
    assert schema["key_groups"]["Core"] == ["action", "grasp_label", "alignment_valid"]

    signals = dataset.signal_payload(
        0, ["action", "control_age_ns", "not_a_key"], 0, 3
    )
    assert signals["series"]["action"][0]["values"] == pytest.approx([0, 0.25, 0])
    assert signals["series"]["control_age_ns"][0]["name"] == "control_age_ms"
    assert signals["series"]["control_age_ns"][0]["values"] == pytest.approx([2, 2, 2])
    assert signals["skipped"] == ["not_a_key"]

    event_types = [event["type"] for event in dataset.events(0)]
    assert "validity" in event_types
    assert "grasp" in event_types

    frame = dataset.frame(0, 0)
    assert frame.shape == (8, 10, 3)
    assert np.all(frame[:, :, 0] == 255)


def test_viewer_http_api_and_assets_are_read_only(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path / "sample.zarr")
    dataset = GirafDataset(path)
    episode_ends_before = dataset.episode_ends.copy()
    server = create_server(dataset, host="127.0.0.1", port=0, requested_episode=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        with urlopen(f"{base_url}/api/dataset/summary", timeout=3) as response:
            summary = json.load(response)
        assert summary["episode_count"] == 2
        assert summary["requested_episode"] == 1

        with urlopen(f"{base_url}/api/episodes", timeout=3) as response:
            episodes = json.load(response)
        assert [item["length"] for item in episodes["episodes"]] == [3, 2]

        with urlopen(f"{base_url}/api/episode/0/frame?idx=1", timeout=3) as response:
            encoded_frame = response.read()
            assert response.headers.get_content_type() == "image/jpeg"
        decoded = cv2.imdecode(np.frombuffer(encoded_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (8, 10, 3)

        with urlopen(f"{base_url}/", timeout=3) as response:
            index_html = response.read().decode("utf-8")
        assert "GIRAF Dataset Viewer" in index_html

        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{base_url}/api/episode/0/delete", timeout=3)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    reopened = zarr.open_group(str(path), mode="r")
    np.testing.assert_array_equal(reopened["meta/episode_ends"][:], episode_ends_before)


def test_viewer_rejects_non_giraf_zarr(tmp_path: Path) -> None:
    path = tmp_path / "other.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.require_group("data")
    meta = root.require_group("meta")
    meta.array("episode_ends", np.array([], dtype=np.int64), chunks=(1,))

    with pytest.raises(DatasetFormatError, match="schema_version"):
        GirafDataset(path)
