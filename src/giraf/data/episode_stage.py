"""Per-episode staging: bounded in-memory batch, on-disk Zarr, optional MP4."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import zarr

from .config import CollectorConfig
from .replay_buffer import ReplayBufferWriter, disk_compressor, git_revision
from .schema import SCHEMA_VERSION, TIME_DATA_KEYS

TIMING_KEYS = ("camera_receive_latency_ns", "control_age_ns", "motor_age_ns")


class EpisodeStage:
    """Bounded in-memory batch plus per-episode on-disk staging area."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        start_wall_time_ns: int,
        start_monotonic_ns: int,
    ) -> None:
        self.config = config
        self.start_wall_time_ns = int(start_wall_time_ns)
        self.start_monotonic_ns = int(start_monotonic_ns)
        self.token = uuid.uuid4().hex
        self.directory = config.dataset.output_dir / ".partial" / self.token
        self.directory.mkdir(parents=True, exist_ok=False)
        self.zarr_path = self.directory / "episode.zarr"
        self.group = zarr.open_group(str(self.zarr_path), mode="w")
        self.data = self.group.require_group("data")
        self.batch: dict[str, list[np.ndarray]] = {key: [] for key in TIME_DATA_KEYS}
        self.length = 0
        self.valid_steps = 0
        self.invalid_steps = 0
        self.video_container = None
        self.video_stream = None
        self.video_path = self.directory / "camera.mp4"
        if config.dataset.save_raw_video:
            self._start_video()

    def _start_video(self) -> None:
        camera = self.config.camera
        self.video_container = av.open(str(self.video_path), mode="w")
        self.video_stream = self.video_container.add_stream(
            self.config.dataset.raw_video_codec,
            rate=int(round(camera.fps)),
        )
        self.video_stream.width = camera.width
        self.video_stream.height = camera.height
        self.video_stream.pix_fmt = "yuv420p"
        self.video_stream.options = {"crf": str(self.config.dataset.raw_video_crf)}

    def _write_video(self, rgb: np.ndarray) -> None:
        if self.video_stream is None or self.video_container is None:
            return
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.video_stream.encode(frame):
            self.video_container.mux(packet)

    def append(self, sample: dict[str, np.ndarray]) -> None:
        source_rgb = np.asarray(sample["camera_rgb_source"], dtype=np.uint8)
        self._write_video(source_rgb)
        width, height = self.config.dataset.resize_dim
        resized = cv2.resize(source_rgb, (width, height), interpolation=cv2.INTER_AREA)
        timestamp_ns = int(sample["timestamp_ns"])
        control_timestamp_ns = int(sample["control_timestamp_ns"])
        motor_timestamp_ns = int(sample["motor_timestamp_ns"])
        grasp = np.uint8(sample["grasp"])
        action = np.concatenate(
            (
                np.asarray(sample["task_twist"], dtype=np.float32),
                np.asarray([grasp], dtype=np.float32),
            )
        )
        converted = {
            "timestamp": np.float64(timestamp_ns / 1_000_000_000.0),
            "timestamp_ns": np.int64(timestamp_ns),
            "camera_rgb": resized.astype(np.uint8, copy=False),
            "grasp_label": grasp,
            "action": action,
            "state": np.asarray(sample["state"], dtype=np.float32),
            "joint_velocity_command": np.asarray(
                sample["joint_velocity_command"], dtype=np.float32
            ),
            "joint_position_command": np.asarray(
                sample["joint_position_command"], dtype=np.float32
            ),
            "can_position_target": np.asarray(
                sample["can_position_target"], dtype=np.float32
            ),
            "dynamixel_target_ticks": np.asarray(
                sample["dynamixel_target_ticks"], dtype=np.int32
            ),
            "tracking": np.uint8(sample["tracking"]),
            "clutch": np.uint8(sample["clutch"]),
            "motor_command_accepted": np.uint8(sample["motor_command_accepted"]),
            "alignment_valid": np.uint8(sample["alignment_valid"]),
            "camera_sequence_num": np.int64(sample["camera_sequence_num"]),
            "camera_device_timestamp_ns": np.int64(
                sample["camera_device_timestamp_ns"]
            ),
            "camera_receive_timestamp_ns": np.int64(
                sample["camera_receive_timestamp_ns"]
            ),
            "control_timestamp_ns": np.int64(control_timestamp_ns),
            "motor_timestamp_ns": np.int64(motor_timestamp_ns),
            "camera_receive_latency_ns": np.int64(
                int(sample["camera_receive_timestamp_ns"]) - timestamp_ns
            ),
            "control_age_ns": np.int64(timestamp_ns - control_timestamp_ns),
            "motor_age_ns": np.int64(timestamp_ns - motor_timestamp_ns),
        }
        for key in TIME_DATA_KEYS:
            self.batch[key].append(np.asarray(converted[key]))
        self.length += 1
        if bool(converted["alignment_valid"]):
            self.valid_steps += 1
        else:
            self.invalid_steps += 1
        if len(self.batch["timestamp_ns"]) >= self.config.dataset.saver_batch_size:
            self.flush()

    def flush(self) -> None:
        n_items = len(self.batch["timestamp_ns"])
        if n_items == 0:
            return
        compressor = disk_compressor()
        for key in TIME_DATA_KEYS:
            values = np.stack(self.batch[key], axis=0)
            if key not in self.data:
                chunks = (self.config.dataset.zarr_chunk_length,) + values.shape[1:]
                self.data.create_dataset(
                    key,
                    shape=(0,) + values.shape[1:],
                    chunks=chunks,
                    dtype=values.dtype,
                    compressor=compressor,
                )
            array = self.data[key]
            old_length = int(array.shape[0])
            array.resize((old_length + n_items,) + array.shape[1:])
            array[old_length:] = values
            self.batch[key].clear()

    def close(self) -> None:
        self.flush()
        if self.video_stream is not None and self.video_container is not None:
            for packet in self.video_stream.encode():
                self.video_container.mux(packet)
            self.video_container.close()
            self.video_stream = None
            self.video_container = None

    def timing_statistics(self) -> dict[str, Any]:
        self.flush()
        statistics: dict[str, Any] = {}
        for key in TIMING_KEYS:
            values = np.asarray(self.data[key][:], dtype=np.int64)
            statistics[key] = {
                "min": int(np.min(values)),
                "p50": int(np.percentile(values, 50)),
                "p95": int(np.percentile(values, 95)),
                "max": int(np.max(values)),
            }
        timestamps = np.asarray(self.data["timestamp_ns"][:], dtype=np.int64)
        sequences = np.asarray(self.data["camera_sequence_num"][:], dtype=np.int64)
        statistics["duration_s"] = (
            float((timestamps[-1] - timestamps[0]) / 1_000_000_000)
            if len(timestamps) > 1
            else 0.0
        )
        statistics["camera_sequence_gaps"] = int(
            np.count_nonzero(np.diff(sequences) != 1)
        )
        return statistics

    def commit(self, writer: ReplayBufferWriter) -> dict[str, Any]:
        self.close()
        expected_episode_index = writer.n_episodes
        final_dir = self.config.video_dir / str(expected_episode_index)
        if final_dir.exists():
            raise FileExistsError(
                f"episode artifact directory already exists: {final_dir}"
            )
        final_dir.mkdir(parents=True, exist_ok=False)
        metadata = {
            "episode_index": expected_episode_index,
            "num_steps": self.length,
            "valid_steps": self.valid_steps,
            "invalid_steps": self.invalid_steps,
            "start_wall_time_ns": self.start_wall_time_ns,
            "start_monotonic_ns": self.start_monotonic_ns,
            "schema_version": SCHEMA_VERSION,
            "raw_100hz_retained": False,
            "git_revision": git_revision(),
            "collector_config": self.config.as_dict(),
            "timing": self.timing_statistics(),
        }
        try:
            if self.video_path.exists():
                self.video_path.replace(final_dir / "camera.mp4")
            (final_dir / "episode.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            episode_index, episode_length = writer.append_stage(
                self.group,
                start_wall_time_ns=self.start_wall_time_ns,
                start_monotonic_ns=self.start_monotonic_ns,
                valid_steps=self.valid_steps,
                invalid_steps=self.invalid_steps,
            )
            if episode_index != expected_episode_index or episode_length != self.length:
                raise RuntimeError("ReplayBuffer commit returned inconsistent metadata")
        except BaseException:
            # episode_ends is the commit marker, so this removes any partially
            # extended arrays when append_stage fails before the commit.
            writer.recover_uncommitted_tail()
            moved_video = final_dir / "camera.mp4"
            if moved_video.exists():
                moved_video.replace(self.video_path)
            shutil.rmtree(final_dir, ignore_errors=True)
            raise
        try:
            shutil.rmtree(self.directory)
        except OSError:
            # The committed ReplayBuffer and final artifacts are authoritative;
            # a stale staging directory can be inspected or removed later.
            pass
        return metadata

    def reject(self, reason: str) -> Path:
        self.close()
        rejected = self.config.dataset.output_dir / "rejected"
        rejected.mkdir(parents=True, exist_ok=True)
        destination = rejected / self.token
        self.directory.replace(destination)
        (destination / "rejection.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "num_steps": self.length,
                    "start_wall_time_ns": self.start_wall_time_ns,
                    "start_monotonic_ns": self.start_monotonic_ns,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return destination
