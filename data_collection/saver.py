"""Dedicated episode saver and Diffusion Policy-compatible Zarr writer."""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .config import CollectorConfig
from .schema import ACTION_FIELDS, JOINT_FIELDS, SCHEMA_VERSION, STATE_FIELDS
from .shared_memory import SharedMemoryRingBuffer

TIME_DATA_KEYS = (
    "timestamp",
    "timestamp_ns",
    "camera_rgb",
    "grasp_label",
    "action",
    "state",
    "joint_velocity_command",
    "joint_position_command",
    "can_position_target",
    "dynamixel_target_ticks",
    "tracking",
    "clutch",
    "motor_command_accepted",
    "alignment_valid",
    "camera_sequence_num",
    "camera_device_timestamp_ns",
    "camera_receive_timestamp_ns",
    "control_timestamp_ns",
    "motor_timestamp_ns",
    "camera_receive_latency_ns",
    "control_age_ns",
    "motor_age_ns",
)

EPISODE_META_KEYS = (
    "episode_start_wall_time_ns",
    "episode_start_monotonic_ns",
    "episode_valid_steps",
    "episode_invalid_steps",
)


def _zarr_modules():
    try:
        import numcodecs
        import zarr
    except ModuleNotFoundError as exc:
        raise RuntimeError("data collection requires zarr<3 and numcodecs") from exc
    major = int(str(zarr.__version__).split(".", 1)[0])
    if major >= 3:
        raise RuntimeError("Diffusion Policy compatibility requires zarr<3")
    return zarr, numcodecs


def _disk_compressor():
    _zarr, numcodecs = _zarr_modules()
    return numcodecs.Blosc(
        cname="zstd",
        clevel=5,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )


def _git_revision() -> str:
    repository = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


class ReplayBufferWriter:
    """Append staged episodes using Diffusion Policy's data/meta layout."""

    def __init__(self, config: CollectorConfig) -> None:
        zarr, _numcodecs = _zarr_modules()
        self.config = config
        config.dataset.output_dir.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(config.zarr_path), mode="a")
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")
        if "episode_ends" not in self.meta:
            self.meta.create_dataset(
                "episode_ends",
                shape=(0,),
                chunks=(1024,),
                dtype=np.int64,
                compressor=None,
            )
        self._set_schema_attributes()
        self.recover_uncommitted_tail()

    def _set_schema_attributes(self) -> None:
        attrs = self.root.attrs
        existing = attrs.get("schema_version")
        if existing is not None and existing != SCHEMA_VERSION:
            raise RuntimeError(
                f"dataset schema is {existing!r}, expected {SCHEMA_VERSION!r}"
            )
        expected = {
            "schema_version": SCHEMA_VERSION,
            "action_fields": list(ACTION_FIELDS),
            "joint_fields": list(JOINT_FIELDS),
            "state_fields": list(STATE_FIELDS),
            "state_semantics": "command-derived; not measured hardware feedback",
            "grasp_semantics": "operator command; not contact sensing",
            "image_layout": "THWC RGB uint8",
            "aligned_hz": float(self.config.dataset.aligned_hz),
            "resize_dim": list(self.config.dataset.resize_dim),
            "resize_mode": self.config.dataset.resize_mode,
            "motor_command_accepted_semantics": (
                "host dispatch calls returned successfully; not hardware feedback"
            ),
        }
        for key in ("aligned_hz", "resize_dim", "resize_mode"):
            if key in attrs and attrs[key] != expected[key]:
                raise RuntimeError(
                    f"dataset attribute {key}={attrs[key]!r} does not match "
                    f"collector setting {expected[key]!r}"
                )
        attrs.update(expected)
        attrs["last_collector_config"] = self.config.as_dict()
        revision = _git_revision()
        revisions = list(attrs.get("git_revisions", []))
        if revision not in revisions:
            revisions.append(revision)
        attrs["git_revisions"] = revisions

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def n_episodes(self) -> int:
        return int(self.episode_ends.shape[0])

    @property
    def n_steps(self) -> int:
        if self.n_episodes == 0:
            return 0
        return int(self.episode_ends[-1])

    def recover_uncommitted_tail(self) -> None:
        """Truncate arrays not covered by the commit-marker episode ends."""

        committed_steps = self.n_steps
        for _name, array in self.data.arrays():
            if array.shape[0] < committed_steps:
                raise RuntimeError("ReplayBuffer data is shorter than episode_ends")
            if array.shape[0] > committed_steps:
                array.resize((committed_steps,) + array.shape[1:])
        committed_episodes = self.n_episodes
        for key in EPISODE_META_KEYS:
            if key in self.meta:
                array = self.meta[key]
                if array.shape[0] < committed_episodes:
                    raise RuntimeError(f"metadata {key} is shorter than episode_ends")
                if array.shape[0] > committed_episodes:
                    array.resize((committed_episodes,))

    def append_stage(
        self,
        stage_group,
        *,
        start_wall_time_ns: int,
        start_monotonic_ns: int,
        valid_steps: int,
        invalid_steps: int,
    ) -> tuple[int, int]:
        stage_data = stage_group["data"]
        keys = set(stage_data.array_keys())
        if keys != set(TIME_DATA_KEYS):
            missing = set(TIME_DATA_KEYS) - keys
            extra = keys - set(TIME_DATA_KEYS)
            raise RuntimeError(
                f"staged data keys differ from schema: missing={missing}, extra={extra}"
            )
        episode_length = int(stage_data["timestamp_ns"].shape[0])
        if episode_length <= 0:
            raise ValueError("cannot commit an empty episode")
        for key in TIME_DATA_KEYS:
            if stage_data[key].shape[0] != episode_length:
                raise RuntimeError(f"staged array {key} has a different time length")

        current = self.n_steps
        new_length = current + episode_length
        compressor = _disk_compressor()

        # Validate every existing array before mutating any of them.
        for key in TIME_DATA_KEYS:
            source = stage_data[key]
            if key in self.data:
                target = self.data[key]
                if target.shape[1:] != source.shape[1:] or target.dtype != source.dtype:
                    raise RuntimeError(
                        f"existing ReplayBuffer schema mismatch for {key}"
                    )

        for key in TIME_DATA_KEYS:
            source = stage_data[key]
            if key not in self.data:
                chunks = (self.config.dataset.zarr_chunk_length,) + source.shape[1:]
                self.data.create_dataset(
                    key,
                    shape=(current,) + source.shape[1:],
                    chunks=chunks,
                    dtype=source.dtype,
                    compressor=compressor,
                )
            target = self.data[key]
            target.resize((new_length,) + target.shape[1:])
            copy_step = max(1, int(source.chunks[0]))
            for start in range(0, episode_length, copy_step):
                stop = min(episode_length, start + copy_step)
                target[current + start : current + stop] = source[start:stop]

        episode_meta = {
            "episode_start_wall_time_ns": np.int64(start_wall_time_ns),
            "episode_start_monotonic_ns": np.int64(start_monotonic_ns),
            "episode_valid_steps": np.int64(valid_steps),
            "episode_invalid_steps": np.int64(invalid_steps),
        }
        episode_index = self.n_episodes
        for key, value in episode_meta.items():
            if key not in self.meta:
                self.meta.create_dataset(
                    key,
                    shape=(episode_index,),
                    chunks=(1024,),
                    dtype=np.int64,
                    compressor=None,
                )
            target = self.meta[key]
            target.resize((episode_index + 1,))
            target[episode_index] = value

        # Commit marker: written only after every time-major and episode-level
        # array is durable enough for restart recovery.
        self.episode_ends.resize((episode_index + 1,))
        self.episode_ends[episode_index] = np.int64(new_length)
        return episode_index, episode_length


class EpisodeStage:
    """Bounded in-memory batch plus per-episode on-disk staging area."""

    def __init__(
        self,
        config: CollectorConfig,
        *,
        start_wall_time_ns: int,
        start_monotonic_ns: int,
    ) -> None:
        zarr, _numcodecs = _zarr_modules()
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
        try:
            import av
        except ModuleNotFoundError as exc:
            raise RuntimeError("raw MP4 recording requires PyAV") from exc
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
        import av

        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.video_stream.encode(frame):
            self.video_container.mux(packet)

    def append(self, sample: dict[str, np.ndarray]) -> None:
        import cv2

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
        compressor = _disk_compressor()
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
        for key in (
            "camera_receive_latency_ns",
            "control_age_ns",
            "motor_age_ns",
        ):
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
            "git_revision": _git_revision(),
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


class SaverProcess(mp.Process):
    """Drain aligned shared memory and own every on-disk dataset write."""

    def __init__(
        self,
        config: CollectorConfig,
        aligned_ring: SharedMemoryRingBuffer,
    ) -> None:
        super().__init__(name="dataset-saver")
        self.config = config
        self.aligned_ring = aligned_ring
        self.ready_event = mp.Event()
        self.parent_connection, self.child_connection = mp.Pipe(duplex=True)
        self._request_id = 0

    def run(self) -> None:
        stage: EpisodeStage | None = None
        cursor = self.aligned_ring.count
        try:
            writer = ReplayBufferWriter(self.config)
            self.config.video_dir.mkdir(parents=True, exist_ok=True)
            self.ready_event.set()
            running = True
            while running:
                if self.child_connection.poll(0.005):
                    command = self.child_connection.recv()
                    request_id = command["request_id"]
                    operation = command["operation"]
                    try:
                        if operation == "start":
                            if stage is not None:
                                raise RuntimeError("an episode is already active")
                            cursor = int(command["start_count"])
                            stage = EpisodeStage(
                                self.config,
                                start_wall_time_ns=int(command["start_wall_time_ns"]),
                                start_monotonic_ns=int(command["start_monotonic_ns"]),
                            )
                            result: dict[str, Any] = {"started": True}
                        elif operation == "stop":
                            if stage is None:
                                raise RuntimeError("no episode is active")
                            cursor = self._drain(
                                stage, cursor, int(command["end_count"])
                            )
                            if stage.length == 0:
                                path = stage.reject("empty episode")
                                stage = None
                                result = {
                                    "rejected": True,
                                    "reason": "empty episode",
                                    "rejected_path": str(path),
                                }
                            else:
                                result = stage.commit(writer)
                                stage = None
                        elif operation == "abort":
                            if stage is not None:
                                path = stage.reject(str(command["reason"]))
                                stage = None
                                result = {"rejected_path": str(path)}
                            else:
                                result = {"rejected_path": None}
                        elif operation == "shutdown":
                            if stage is not None:
                                stage.reject("saver shutdown with active episode")
                                stage = None
                            result = {"shutdown": True}
                            running = False
                        else:
                            raise ValueError(f"unknown saver operation {operation!r}")
                        self.child_connection.send(
                            {"request_id": request_id, "ok": True, "result": result}
                        )
                    except BaseException as exc:
                        if stage is not None and operation in {"start", "stop"}:
                            try:
                                stage.reject(f"{type(exc).__name__}: {exc}")
                            finally:
                                stage = None
                        self.child_connection.send(
                            {
                                "request_id": request_id,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if stage is not None:
                    cursor = self._drain(stage, cursor, self.aligned_ring.count)
        except BaseException:
            if stage is not None:
                try:
                    stage.reject("saver process failure")
                except BaseException:
                    pass
            try:
                self.child_connection.send(
                    {
                        "request_id": -1,
                        "ok": False,
                        "error": traceback.format_exc(),
                    }
                )
            except (BrokenPipeError, EOFError, OSError):
                pass
        finally:
            self.ready_event.clear()

    def _drain(self, stage: EpisodeStage, cursor: int, target: int) -> int:
        while cursor < target:
            stop = min(target, cursor + self.aligned_ring.get_max_k)
            batch = self.aligned_ring.get_range(cursor, stop)
            for index in range(stop - cursor):
                stage.append({key: value[index] for key, value in batch.items()})
            cursor = stop
        return cursor

    def start_wait(self, timeout: float = 15.0) -> None:
        if not self.ready_event.wait(timeout):
            if self.parent_connection.poll():
                message = self.parent_connection.recv()
                raise RuntimeError(message.get("error", "saver startup failed"))
            raise TimeoutError("dataset saver did not become ready")

    def request(self, operation: str, timeout: float = 30.0, **payload):
        if self.pid is None or self.exitcode is not None:
            raise RuntimeError("dataset saver is not running")
        self._request_id += 1
        request_id = self._request_id
        self.parent_connection.send(
            {"request_id": request_id, "operation": operation, **payload}
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.parent_connection.poll(0.05):
                response = self.parent_connection.recv()
                if response.get("request_id") not in (request_id, -1):
                    continue
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "saver request failed"))
                return response.get("result")
            if self.exitcode is not None:
                raise RuntimeError(f"dataset saver exited with code {self.exitcode}")
        raise TimeoutError(f"saver operation {operation!r} timed out")

    def shutdown(self) -> None:
        if self.pid is None:
            return
        if self.is_alive():
            try:
                self.request("shutdown", timeout=10.0)
            except (RuntimeError, TimeoutError, BrokenPipeError, EOFError, OSError):
                pass
            self.join(timeout=5.0)
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)
