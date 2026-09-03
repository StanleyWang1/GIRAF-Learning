from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("zarr")
import zarr
from fastapi.testclient import TestClient

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.app import create_app
from dataset_visualization.backend.config import AppConfig
from dataset_visualization.backend.services import inference as inference_service
from dataset_visualization.backend.services.inference import (
    InferenceConfigError,
    build_inference_observation,
    load_inference_task_config,
    run_inference_overlay,
)
from dataset_visualization.backend.types import CameraStream, DatasetSummary, EpisodeSchema


def _yaml_line(indent: int, key: str, value: int | None) -> str:
    if value is None:
        return ""
    return f"{' ' * indent}{key}: {value}\n"


def _write_inference_yaml(
    path: Path,
    *,
    predict_horizon: int | None = 3,
    img_obs_horizon: int | None = 4,
    low_dim_obs_horizon: int | None = 4,
    short_range_obs_window: int | None = 4,
    camera_horizon: int | None = 4,
    pos_horizon: int | None = 4,
    rpy_horizon: int | None = 4,
    gripper_horizon: int | None = 4,
    camera_short_dss: int | None = 4,
    camera_long_dss: int | None = 6,
    pos_short_dss: int | None = 3,
    pos_long_dss: int | None = 3,
    rpy_short_dss: int | None = 3,
    rpy_long_dss: int | None = 3,
    gripper_short_dss: int | None = 3,
    gripper_long_dss: int | None = 3,
    action_dss: int | None = 3,
) -> None:
    path.write_text(
        (
            "policy:\n"
            + "  obs_encoder:\n"
            + "    short_range:\n"
            + f"      obs_window: {'null' if short_range_obs_window is None else short_range_obs_window}\n"
            + "task:\n"
            + _yaml_line(2, "inference_time_img_obs_horizon", predict_horizon)
            + _yaml_line(2, "img_obs_horizon", img_obs_horizon)
            + _yaml_line(2, "low_dim_obs_horizon", low_dim_obs_horizon)
            + "  shape_meta:\n"
            + "    obs:\n"
            + "      camera_cam0_rgb:\n"
            + "        shape: [3, 4, 4]\n"
            + _yaml_line(8, "horizon", camera_horizon)
            + _yaml_line(8, "short_dss", camera_short_dss)
            + _yaml_line(8, "long_dss", camera_long_dss)
            + "        type: rgb\n"
            + "        ignore_by_policy: false\n"
            + "      robot0_eef_pos:\n"
            + "        shape: [3]\n"
            + _yaml_line(8, "horizon", pos_horizon)
            + _yaml_line(8, "short_dss", pos_short_dss)
            + _yaml_line(8, "long_dss", pos_long_dss)
            + "        type: low_dim\n"
            + "        ignore_by_policy: false\n"
            + "      robot0_eef_rpy:\n"
            + "        raw_shape: [3]\n"
            + "        shape: [6]\n"
            + _yaml_line(8, "horizon", rpy_horizon)
            + _yaml_line(8, "short_dss", rpy_short_dss)
            + _yaml_line(8, "long_dss", rpy_long_dss)
            + "        type: low_dim\n"
            + "        ignore_by_policy: false\n"
            + "      robot0_gripper_width:\n"
            + "        shape: [1]\n"
            + _yaml_line(8, "horizon", gripper_horizon)
            + _yaml_line(8, "short_dss", gripper_short_dss)
            + _yaml_line(8, "long_dss", gripper_long_dss)
            + "        type: low_dim\n"
            + "        ignore_by_policy: false\n"
            + "    action:\n"
            + "      shape: [7]\n"
            + _yaml_line(6, "dss", action_dss)
        ).strip()
    )


class DummyInferenceAdapter(BaseAdapter):
    format_name = "dummy_inference"

    def __init__(self):
        super().__init__()
        self.length = 12
        frames = np.arange(self.length, dtype=np.float32)
        self._robot0_eef_pos = np.stack([frames, frames + 0.1, frames + 0.2], axis=1)
        self._robot0_eef_rpy = np.stack([frames + 1.0, frames + 2.0, frames + 3.0], axis=1)
        self._robot0_gripper_width = (frames / 10.0).reshape(-1, 1)
        self.rgb_requests: list[tuple[str, int]] = []
        self.signal_requests: list[tuple[str, int, int, int]] = []

    def dataset_summary(self):
        return DatasetSummary("dummy", self.format_name, True, None, 1, self.length)

    def episode_count(self):
        return 1

    def episode_bounds(self, episode_index):
        return (0, self.length)

    def episode_schema(self, episode_index):
        return EpisodeSchema(
            episode_index=0,
            length=self.length,
            start=0,
            end=self.length,
            has_timestamps=False,
            cameras=[CameraStream(stream_id="cam0", rgb_source="zarr", rgb_key="camera_cam0_rgb", rgb_shape=(6, 6, 3))],
        )

    def episode_timestamps(self, episode_index, start, end, stride=1):
        return np.arange(start, end, stride, dtype=np.float64)

    def signal_window(self, episode_index, key, start, end, stride=1):
        self.signal_requests.append((str(key), int(start), int(end), int(stride)))
        data = {
            "robot0_eef_pos": self._robot0_eef_pos,
            "robot0_eef_rpy": self._robot0_eef_rpy,
            "robot0_gripper_width": self._robot0_gripper_width,
        }.get(key)
        if data is None:
            raise KeyError(key)
        return np.asarray(data[start:end:stride])

    def frame_rgb(self, episode_index, stream_id, idx):
        self.rgb_requests.append((str(stream_id), int(idx)))
        return np.full((6, 6, 3), fill_value=int(idx), dtype=np.uint8)

    def frame_depth(self, episode_index, stream_id, idx):
        raise NotImplementedError

    def list_stream_ids(self, episode_index):
        return ["cam0"]

    def has_depth(self, episode_index, stream_id):
        return False

    def graphable_keys(self, episode_index):
        return ["robot0_eef_pos", "robot0_eef_rpy", "robot0_gripper_width"]

    def all_keys(self):
        return ["robot0_eef_pos", "robot0_eef_rpy", "robot0_gripper_width"]


class FakeTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.closed = False

    def request(self, payload):
        self.requests.append(payload)
        if not self.replies:
            raise RuntimeError("no fake reply available")
        return self.replies.pop(0)

    def close(self):
        self.closed = True


def _create_full_zarr_dataset(path: Path):
    root = zarr.open(str(path), mode="w")
    data = root.require_group("data")
    meta = root.require_group("meta")
    frames = np.arange(5, dtype=np.float32)

    rgb = np.stack([np.full((8, 8, 3), fill_value=i, dtype=np.uint8) for i in range(5)], axis=0)
    data.array("camera_cam0_rgb", rgb, chunks=(5, 8, 8, 3), overwrite=True)
    data.array(
        "robot0_eef_pos",
        np.stack([frames, frames + 0.5, frames + 1.0], axis=1).astype(np.float32),
        chunks=(5, 3),
        overwrite=True,
    )
    data.array(
        "robot0_eef_rpy",
        np.stack([frames + 2.0, frames + 3.0, frames + 4.0], axis=1).astype(np.float32),
        chunks=(5, 3),
        overwrite=True,
    )
    data.array("robot0_gripper_width", frames.reshape(-1, 1).astype(np.float32), chunks=(5, 1), overwrite=True)
    meta.array("episode_ends", np.array([5], dtype=np.int64), chunks=(1,), overwrite=True)


def test_build_inference_observation_uses_short_dss_and_numeric_batch_dim(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    task_cfg = load_inference_task_config(str(yaml_path))
    adapter = DummyInferenceAdapter()

    assert task_cfg.predict_horizon == 3
    assert task_cfg.img_horizon == 4
    assert task_cfg.low_dim_horizon == 4
    assert task_cfg.action_downsample_steps == 3

    obs = build_inference_observation(adapter, 0, 10, task_cfg)

    assert len(obs["camera_cam0_rgb"]) == 3
    assert adapter.rgb_requests == [("cam0", 2), ("cam0", 6), ("cam0", 10)]
    assert obs["robot0_eef_pos"].shape == (1, 3, 3)
    np.testing.assert_allclose(obs["robot0_eef_pos"][0, :, 0], np.array([4, 7, 10], dtype=np.float32))
    assert obs["robot0_eef_rpy"].shape == (1, 3, 3)
    assert obs["robot0_gripper_width"].shape == (1, 3, 1)

    short_obs = build_inference_observation(adapter, 0, 10, task_cfg, k=1)
    assert len(short_obs["camera_cam0_rgb"]) == 1
    assert short_obs["robot0_eef_pos"].shape == (1, 1, 3)
    assert short_obs["robot0_gripper_width"].shape == (1, 1, 1)


def test_run_inference_overlay_sequences_wrapped_requests_with_exact_warmup_steps(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1, 10, 20, 30, 0, 0, 0, 0.2],
                    [4, 5, 6, 0, 0, 0, 0.3, 40, 50, 60, 0, 0, 0, 0.4],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=10,
        yaml_path=str(yaml_path),
        server_host="127.0.0.1",
        server_port=8767,
        warmup_steps=4,
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert [req["request_type"] for req in fake_transport.requests] == [
        "reset_memory",
        "update_memory",
        "update_memory",
        "update_memory",
        "update_memory",
        "predict_action",
    ]
    warmup_robot_frames = [
        int(req["obs"]["long"]["robot0_eef_pos"][0, 0]) for req in fake_transport.requests[1:5]
    ]
    warmup_rgb_frames = [frame for stream_id, frame in adapter.rgb_requests[:4] if stream_id == "cam0"]
    assert warmup_robot_frames == [1, 4, 7, 10]
    assert warmup_rgb_frames == [0, 0, 4, 10]
    assert "batch_size" not in fake_transport.requests[1]["obs"]["long"]
    assert fake_transport.requests[-1]["obs"]["short"]["robot0_eef_pos"].shape == (3, 3)
    assert "batch_size" not in fake_transport.requests[-1]["obs"]["short"]
    assert fake_transport.requests[-1]["obs"]["batch_size"] == 1
    assert payload["predict_obs_horizon_used"] == 3
    assert payload["warmup_steps_requested"] == 4
    assert payload["warmup_steps_effective"] == 4
    assert payload["warmup_update_calls_requested"] == 4
    assert payload["warmup_update_calls_effective"] == 4
    assert payload["warmup_update_frames"] == []
    assert len(payload["warmup_update_snapshots"]) == 4
    assert payload["warmup_update_snapshots"][0]["obs_frame_indices"]["camera_cam0_rgb"] == [0]
    assert payload["warmup_update_snapshots"][1]["obs_frame_indices"]["robot0_eef_pos"] == [4]
    assert payload["batch_size_requested"] == 1
    assert payload["batch_count_returned"] == 1
    assert payload["action_downsample_steps"] == 3
    assert payload["action_shape"] == [2, 14]
    assert payload["batch_action_shapes"] == [[2, 14]]
    assert len(payload["batches"]) == 1
    assert payload["robots"]["robot0"]["predicted_pos"][0] == [1.0, 2.0, 3.0]
    assert payload["robots"]["robot1"]["predicted_pos"][1] == [40.0, 50.0, 60.0]
    assert fake_transport.closed is True


def test_run_inference_overlay_supports_batched_predictions(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            {"status": "ok"},
            {"status": "ok"},
            np.asarray(
                [
                    [
                        [1, 2, 3, 0, 0, 0, 0.1],
                        [4, 5, 6, 0, 0, 0, 0.2],
                    ],
                    [
                        [10, 20, 30, 0, 0, 0, 0.3],
                        [40, 50, 60, 0, 0, 0, 0.4],
                    ],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=0,
        yaml_path=str(yaml_path),
        batch_size=2,
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert [req["request_type"] for req in fake_transport.requests] == ["reset_memory", "update_memory", "predict_action"]
    assert fake_transport.requests[1]["obs"]["long"]["robot0_eef_pos"].shape == (1, 3)
    assert fake_transport.requests[2]["obs"]["batch_size"] == 2
    assert payload["batch_size_requested"] == 2
    assert payload["warmup_update_calls_effective"] == 1
    assert payload["batch_count_returned"] == 2
    assert payload["action_downsample_steps"] == 3
    assert payload["action_shape"] == [2, 2, 7]
    assert payload["batch_action_shapes"] == [[2, 7], [2, 7]]
    assert len(payload["batches"]) == 2
    assert payload["batches"][0]["robots"]["robot0"]["predicted_pos"][0] == [1.0, 2.0, 3.0]
    assert payload["batches"][1]["robots"]["robot0"]["predicted_pos"][1] == [40.0, 50.0, 60.0]


def test_run_inference_overlay_train_time_sends_combined_short_and_long_contexts(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(
        yaml_path,
        pos_short_dss=2,
        rpy_short_dss=2,
        gripper_short_dss=2,
    )
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1],
                    [4, 5, 6, 0, 0, 0, 0.2],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=10,
        yaml_path=str(yaml_path),
        warmup_steps=99,
        batch_size=2,
        inference_mode="train_time",
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert [req["request_type"] for req in fake_transport.requests] == ["predict_action_training"]
    request = fake_transport.requests[0]
    assert set(request["obs"].keys()) == {"long", "short", "batch_size"}
    assert request["obs"]["short"]["robot0_eef_pos"].shape == (4, 3)
    assert request["obs"]["long"]["robot0_eef_pos"].shape == (4, 3)
    assert request["obs"]["short"]["robot0_eef_pos"][:, 0].tolist() == [4.0, 6.0, 8.0, 10.0]
    assert request["obs"]["long"]["robot0_eef_pos"][:, 0].tolist() == [1.0, 4.0, 7.0, 10.0]
    assert len(request["obs"]["short"]["camera_cam0_rgb"]) == 4
    assert len(request["obs"]["long"]["camera_cam0_rgb"]) == 4
    assert request["obs"]["batch_size"] == 2
    assert "batch_size" not in request["obs"]["short"]
    assert "batch_size" not in request["obs"]["long"]
    assert payload["inference_mode"] == "train_time"
    assert payload["request_type_sent"] == "predict_action_training"
    assert payload["warmup_steps_requested"] == 0
    assert payload["warmup_steps_effective"] == 0
    assert payload["warmup_update_calls_requested"] == 0
    assert payload["warmup_update_calls_effective"] == 0
    assert payload["warmup_update_snapshots"] == []
    assert payload["predict_obs_horizon_used"] == 4
    assert payload["long_obs_horizon_used"] == 4
    assert payload["long_sampling_details"]["camera_cam0_rgb"]["frame_indices"] == [0, 0, 4, 10]
    assert payload["predict_sampling_details"]["camera_cam0_rgb"]["frame_indices"] == [0, 2, 6, 10]
    assert payload["long_sampling_details"]["robot0_eef_pos"]["frame_indices"] == [1, 4, 7, 10]
    assert payload["predict_sampling_details"]["robot0_eef_pos"]["frame_indices"] == [4, 6, 8, 10]


def test_run_inference_overlay_train_time_prefers_per_key_horizons(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(
        yaml_path,
        img_obs_horizon=9,
        low_dim_obs_horizon=8,
        camera_horizon=4,
        pos_horizon=2,
        rpy_horizon=2,
        gripper_horizon=2,
        pos_short_dss=2,
        rpy_short_dss=2,
        gripper_short_dss=2,
    )
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1],
                    [4, 5, 6, 0, 0, 0, 0.2],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=10,
        yaml_path=str(yaml_path),
        batch_size=2,
        inference_mode="train_time",
        transport_factory=lambda _host, _port: fake_transport,
    )

    request = fake_transport.requests[0]
    assert request["obs"]["long"]["robot0_eef_pos"].shape == (2, 3)
    assert request["obs"]["short"]["robot0_eef_pos"].shape == (2, 3)
    assert request["obs"]["long"]["robot0_eef_pos"][:, 0].tolist() == [7.0, 10.0]
    assert request["obs"]["short"]["robot0_eef_pos"][:, 0].tolist() == [8.0, 10.0]
    assert len(request["obs"]["long"]["camera_cam0_rgb"]) == 4
    assert len(request["obs"]["short"]["camera_cam0_rgb"]) == 4
    assert payload["long_obs_horizon_used"] == 4
    assert payload["predict_obs_horizon_used"] == 4
    assert payload["long_sampling_details"]["camera_cam0_rgb"]["horizon_used"] == 4
    assert payload["predict_sampling_details"]["camera_cam0_rgb"]["horizon_used"] == 4
    assert payload["long_sampling_details"]["robot0_eef_pos"]["horizon_used"] == 2
    assert payload["predict_sampling_details"]["robot0_eef_pos"]["horizon_used"] == 2
    assert payload["long_sampling_details"]["robot0_eef_pos"]["frame_indices"] == [7, 10]
    assert payload["predict_sampling_details"]["robot0_eef_pos"]["frame_indices"] == [8, 10]


def test_run_inference_overlay_train_time_skips_short_context_when_short_range_disabled(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(
        yaml_path,
        short_range_obs_window=None,
        camera_short_dss=None,
        pos_short_dss=None,
        rpy_short_dss=None,
        gripper_short_dss=None,
    )
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1],
                    [4, 5, 6, 0, 0, 0, 0.2],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=10,
        yaml_path=str(yaml_path),
        batch_size=2,
        inference_mode="train_time",
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert [req["request_type"] for req in fake_transport.requests] == ["predict_action_training"]
    request = fake_transport.requests[0]
    assert set(request["obs"].keys()) == {"long", "batch_size"}
    assert request["obs"]["long"]["robot0_eef_pos"].shape == (4, 3)
    assert request["obs"]["batch_size"] == 2
    assert "batch_size" not in request["obs"]["long"]
    assert payload["inference_mode"] == "train_time"
    assert payload["request_type_sent"] == "predict_action_training"
    assert payload["predict_obs_horizon_used"] == 0
    assert payload["long_obs_horizon_used"] == 4
    assert payload["predict_sampling_details"] == {}
    assert payload["long_sampling_details"]["robot0_eef_pos"]["frame_indices"] == [1, 4, 7, 10]
    assert "obs_window is null" in payload["diagnostics"]["message"]


def test_run_inference_overlay_short_only_sends_short_context_only(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1],
                    [4, 5, 6, 0, 0, 0, 0.2],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=10,
        yaml_path=str(yaml_path),
        warmup_steps=99,
        batch_size=2,
        inference_mode="short_only",
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert [req["request_type"] for req in fake_transport.requests] == ["predict_action_no_memory"]
    request = fake_transport.requests[0]
    assert set(request["obs"].keys()) == {"short", "batch_size"}
    assert request["obs"]["short"]["robot0_eef_pos"].shape == (3, 3)
    assert request["obs"]["short"]["robot0_eef_pos"][:, 0].tolist() == [4.0, 7.0, 10.0]
    assert len(request["obs"]["short"]["camera_cam0_rgb"]) == 3
    assert request["obs"]["batch_size"] == 2
    assert payload["inference_mode"] == "short_only"
    assert payload["request_type_sent"] == "predict_action_no_memory"
    assert payload["warmup_steps_requested"] == 0
    assert payload["warmup_steps_effective"] == 0
    assert payload["warmup_update_calls_requested"] == 0
    assert payload["warmup_update_calls_effective"] == 0
    assert payload["warmup_update_snapshots"] == []
    assert payload["predict_obs_horizon_used"] == 3
    assert payload["long_obs_horizon_used"] == 0
    assert payload["long_sampling_details"] == {}
    assert payload["predict_sampling_details"]["robot0_eef_pos"]["frame_indices"] == [4, 7, 10]


def test_run_inference_overlay_expands_warmup_and_clamps_history(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1],
                    [4, 5, 6, 0, 0, 0, 0.2],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=1,
        yaml_path=str(yaml_path),
        warmup_steps=5,
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert [req["request_type"] for req in fake_transport.requests] == [
        "reset_memory",
        "update_memory",
        "update_memory",
        "update_memory",
        "update_memory",
        "update_memory",
        "predict_action",
    ]
    warmup_robot_frames = [
        int(req["obs"]["long"]["robot0_eef_pos"][0, 0]) for req in fake_transport.requests[1:6]
    ]
    assert warmup_robot_frames == [0, 0, 0, 0, 1]
    assert payload["warmup_steps_requested"] == 5
    assert payload["warmup_steps_effective"] == 5
    assert payload["warmup_update_calls_effective"] == 5
    assert payload["warmup_update_frames"] == [0, 0, 0, 0, 1]


def test_run_inference_overlay_clamps_batch_size_to_max(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            {"status": "ok"},
            {"status": "ok"},
            np.asarray(
                [
                    [1, 2, 3, 0, 0, 0, 0.1],
                    [4, 5, 6, 0, 0, 0, 0.2],
                ],
                dtype=np.float32,
            ),
        ]
    )

    payload = run_inference_overlay(
        adapter=adapter,
        episode_index=0,
        frame_index=0,
        yaml_path=str(yaml_path),
        batch_size=999,
        transport_factory=lambda _host, _port: fake_transport,
    )

    assert fake_transport.requests[2]["obs"]["batch_size"] == 32
    assert payload["batch_size_requested"] == 32
    assert payload["action_downsample_steps"] == 3


def test_run_inference_overlay_rejects_unsupported_action_shape(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(replies=[{"status": "ok"}, {"status": "ok"}, np.zeros((3, 8), dtype=np.float32)])

    with pytest.raises(ValueError, match="unsupported predicted action shape"):
        run_inference_overlay(
            adapter=adapter,
            episode_index=0,
            frame_index=0,
            yaml_path=str(yaml_path),
            warmup_steps=1,
            transport_factory=lambda _host, _port: fake_transport,
        )


@pytest.mark.parametrize(
    ("write_kwargs", "match"),
    [
        ({"camera_long_dss": None}, "missing valid long_dss"),
        ({"predict_horizon": None}, "invalid task.inference_time_img_obs_horizon"),
        ({"img_obs_horizon": None}, "invalid task.img_obs_horizon"),
        ({"low_dim_obs_horizon": None}, "invalid task.low_dim_obs_horizon"),
        ({"action_dss": None}, "invalid task.shape_meta.action.dss"),
    ],
)
def test_load_inference_task_config_validates_required_fields(
    tmp_path: Path,
    write_kwargs: dict[str, int | None],
    match: str,
):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path, **write_kwargs)

    with pytest.raises(InferenceConfigError, match=match):
        load_inference_task_config(str(yaml_path))


def test_load_inference_task_config_allows_distinct_image_and_low_dim_horizons(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path, img_obs_horizon=4, low_dim_obs_horizon=5)

    task_cfg = load_inference_task_config(str(yaml_path))

    assert task_cfg.img_horizon == 4
    assert task_cfg.low_dim_horizon == 5


@pytest.mark.parametrize("inference_mode", ["stateful_memory", "short_only"])
def test_run_inference_overlay_requires_short_dss_outside_train_time(tmp_path: Path, inference_mode: str):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(
        yaml_path,
        camera_short_dss=None,
        pos_short_dss=None,
        rpy_short_dss=None,
        gripper_short_dss=None,
    )
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(replies=[np.zeros((2, 7), dtype=np.float32)])

    with pytest.raises(InferenceConfigError, match="missing valid short_dss"):
        run_inference_overlay(
            adapter=adapter,
            episode_index=0,
            frame_index=10,
            yaml_path=str(yaml_path),
            inference_mode=inference_mode,
            transport_factory=lambda _host, _port: fake_transport,
        )

    assert fake_transport.requests == []
    assert fake_transport.closed is False


def test_inference_api_returns_overlay_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zpath = tmp_path / "inference.zarr"
    yaml_path = tmp_path / "config.yaml"
    _create_full_zarr_dataset(zpath)
    _write_inference_yaml(yaml_path, predict_horizon=2, img_obs_horizon=3, low_dim_obs_horizon=3)

    fake_transport = FakeTransport(
        replies=[
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            np.asarray(
                [
                    [0.1, 0.2, 0.3, 0, 0, 0, 0.01],
                    [0.4, 0.5, 0.6, 0, 0, 0, 0.02],
                ],
                dtype=np.float32,
            ),
        ]
    )

    monkeypatch.setattr(
        inference_service,
        "create_zmq_transport",
        lambda _host, _port: fake_transport,
    )

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

    response = client.post(
        "/api/inference/run",
        json={
            "episode_index": 0,
            "frame_index": 1,
            "yaml_path": str(yaml_path),
            "server_host": "localhost",
            "server_port": 8767,
            "warmup_steps": 2,
            "batch_size": 1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["inference_mode"] == "stateful_memory"
    assert payload["request_type_sent"] == "predict_action"
    assert payload["episode_index"] == 0
    assert payload["frame_index"] == 1
    assert payload["batch_size_requested"] == 1
    assert payload["warmup_update_calls_effective"] == 2
    assert payload["batch_count_returned"] == 1
    assert payload["action_downsample_steps"] == 3
    assert payload["action_shape"] == [2, 7]
    assert payload["robots"]["robot0"]["predicted_pos"][0] == pytest.approx([0.1, 0.2, 0.3])
    assert [req["request_type"] for req in fake_transport.requests] == [
        "reset_memory",
        "update_memory",
        "update_memory",
        "predict_action",
    ]
    assert "batch_size" not in fake_transport.requests[-1]["obs"]["short"]
    assert fake_transport.requests[-1]["obs"]["batch_size"] == 1

    missing_yaml_response = client.post(
        "/api/inference/run",
        json={
            "episode_index": 0,
            "frame_index": 1,
            "yaml_path": str(tmp_path / "missing.yaml"),
            "server_host": "localhost",
            "server_port": 8767,
            "warmup_steps": 1,
            "batch_size": 1,
        },
    )
    assert missing_yaml_response.status_code == 400
    assert "inference yaml not found" in missing_yaml_response.json()["detail"]


def test_inference_api_accepts_train_time_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zpath = tmp_path / "inference_train_time.zarr"
    yaml_path = tmp_path / "config.yaml"
    _create_full_zarr_dataset(zpath)
    _write_inference_yaml(yaml_path, predict_horizon=2, img_obs_horizon=3, low_dim_obs_horizon=3)

    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [0.1, 0.2, 0.3, 0, 0, 0, 0.01],
                    [0.4, 0.5, 0.6, 0, 0, 0, 0.02],
                ],
                dtype=np.float32,
            ),
        ]
    )

    monkeypatch.setattr(
        inference_service,
        "create_zmq_transport",
        lambda _host, _port: fake_transport,
    )

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

    response = client.post(
        "/api/inference/run",
        json={
            "episode_index": 0,
            "frame_index": 1,
            "yaml_path": str(yaml_path),
            "server_host": "localhost",
            "server_port": 8767,
            "warmup_steps": 5,
            "batch_size": 1,
            "inference_mode": "train_time",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_mode"] == "train_time"
    assert payload["request_type_sent"] == "predict_action_training"
    assert payload["warmup_steps_requested"] == 0
    assert payload["warmup_update_calls_effective"] == 0
    assert payload["long_obs_horizon_used"] == 3
    assert [req["request_type"] for req in fake_transport.requests] == ["predict_action_training"]
    assert set(fake_transport.requests[0]["obs"].keys()) == {"long", "short", "batch_size"}


def test_inference_api_train_time_allows_long_only_when_short_range_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    zpath = tmp_path / "inference_train_time_long_only.zarr"
    yaml_path = tmp_path / "config.yaml"
    _create_full_zarr_dataset(zpath)
    _write_inference_yaml(
        yaml_path,
        predict_horizon=2,
        img_obs_horizon=3,
        low_dim_obs_horizon=3,
        short_range_obs_window=None,
        camera_short_dss=None,
        pos_short_dss=None,
        rpy_short_dss=None,
        gripper_short_dss=None,
    )

    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [0.1, 0.2, 0.3, 0, 0, 0, 0.01],
                    [0.4, 0.5, 0.6, 0, 0, 0, 0.02],
                ],
                dtype=np.float32,
            ),
        ]
    )

    monkeypatch.setattr(
        inference_service,
        "create_zmq_transport",
        lambda _host, _port: fake_transport,
    )

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

    response = client.post(
        "/api/inference/run",
        json={
            "episode_index": 0,
            "frame_index": 1,
            "yaml_path": str(yaml_path),
            "server_host": "localhost",
            "server_port": 8767,
            "batch_size": 2,
            "inference_mode": "train_time",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_mode"] == "train_time"
    assert payload["request_type_sent"] == "predict_action_training"
    assert payload["predict_obs_horizon_used"] == 0
    assert payload["long_obs_horizon_used"] == 3
    assert payload["predict_sampling_details"] == {}
    assert [req["request_type"] for req in fake_transport.requests] == ["predict_action_training"]
    assert set(fake_transport.requests[0]["obs"].keys()) == {"long", "batch_size"}


def test_inference_api_accepts_short_only_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    zpath = tmp_path / "inference_no_memory.zarr"
    yaml_path = tmp_path / "config.yaml"
    _create_full_zarr_dataset(zpath)
    _write_inference_yaml(yaml_path, predict_horizon=2, img_obs_horizon=3, low_dim_obs_horizon=3)

    fake_transport = FakeTransport(
        replies=[
            np.asarray(
                [
                    [0.1, 0.2, 0.3, 0, 0, 0, 0.01],
                    [0.4, 0.5, 0.6, 0, 0, 0, 0.02],
                ],
                dtype=np.float32,
            ),
        ]
    )

    monkeypatch.setattr(
        inference_service,
        "create_zmq_transport",
        lambda _host, _port: fake_transport,
    )

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

    response = client.post(
        "/api/inference/run",
        json={
            "episode_index": 0,
            "frame_index": 1,
            "yaml_path": str(yaml_path),
            "server_host": "localhost",
            "server_port": 8767,
            "warmup_steps": 5,
            "batch_size": 1,
            "inference_mode": "short_only",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_mode"] == "short_only"
    assert payload["request_type_sent"] == "predict_action_no_memory"
    assert payload["warmup_steps_requested"] == 0
    assert payload["warmup_update_calls_effective"] == 0
    assert payload["long_obs_horizon_used"] == 0
    assert [req["request_type"] for req in fake_transport.requests] == ["predict_action_no_memory"]
    assert set(fake_transport.requests[0]["obs"].keys()) == {"short", "batch_size"}


def test_run_inference_overlay_surfaces_server_traceback_as_runtime_error(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path)
    adapter = DummyInferenceAdapter()
    fake_transport = FakeTransport(
        replies=[
            {"status": "ok"},
            {"status": "ok"},
            "Traceback (most recent call last):\n  RuntimeError: bad obs shape",
        ]
    )

    with pytest.raises(RuntimeError, match="inference server returned an error"):
        run_inference_overlay(
            adapter=adapter,
            episode_index=0,
            frame_index=0,
            yaml_path=str(yaml_path),
            warmup_steps=1,
            transport_factory=lambda _host, _port: fake_transport,
        )


def test_load_inference_task_config_refreshes_when_yaml_changes(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    _write_inference_yaml(yaml_path, predict_horizon=2)

    first = load_inference_task_config(str(yaml_path))
    assert first.predict_horizon == 2

    _write_inference_yaml(yaml_path, predict_horizon=4)
    second = load_inference_task_config(str(yaml_path))
    assert second.predict_horizon == 4
