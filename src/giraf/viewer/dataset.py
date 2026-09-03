"""GIRAF-aware, read-only access to a replay-buffer Zarr dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr

TIMESTAMP_KEYS = {
    "timestamp",
    "timestamp_ns",
    "camera_device_timestamp_ns",
    "camera_receive_timestamp_ns",
    "control_timestamp_ns",
    "motor_timestamp_ns",
}

MILLISECOND_KEYS = {
    "camera_receive_latency_ns",
    "control_age_ns",
    "motor_age_ns",
}

GROUP_ORDER = ("Core", "State", "Commands", "Health", "Timing", "All")


class DatasetFormatError(ValueError):
    """Raised when a path is not a supported GIRAF replay-buffer dataset."""


def _plain(value: Any) -> Any:
    """Convert Zarr/NumPy attribute values to JSON-friendly Python objects."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class GirafDataset:
    """Read a GIRAF replay buffer without importing collection or video code."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"dataset does not exist: {self.path}")

        try:
            self.root = zarr.open_group(str(self.path), mode="r")
        except Exception as exc:
            raise DatasetFormatError(f"unable to open Zarr dataset: {exc}") from exc

        if "data" not in self.root or "meta" not in self.root:
            raise DatasetFormatError("expected Zarr groups data/ and meta/")
        self.data = self.root["data"]
        self.meta = self.root["meta"]
        if "episode_ends" not in self.meta:
            raise DatasetFormatError("expected meta/episode_ends")

        self.attrs = _plain(dict(self.root.attrs))
        schema_version = self.attrs.get("schema_version")
        if schema_version != "giraf-replay-v1":
            raise DatasetFormatError(
                f"expected schema_version='giraf-replay-v1', found {schema_version!r}"
            )

        self.episode_ends = np.asarray(self.meta["episode_ends"][:], dtype=np.int64)
        if self.episode_ends.ndim != 1:
            raise DatasetFormatError("meta/episode_ends must be one-dimensional")
        if self.episode_ends.size and (
            np.any(self.episode_ends <= 0) or np.any(np.diff(self.episode_ends) <= 0)
        ):
            raise DatasetFormatError("meta/episode_ends must be strictly increasing")

        self.total_steps = int(self.episode_ends[-1]) if self.episode_ends.size else 0
        self.keys = sorted(self.data.array_keys())
        self.issues: list[str] = []
        self.aligned_keys: list[str] = []
        for key in self.keys:
            array = self.data[key]
            if array.ndim < 1 or int(array.shape[0]) != self.total_steps:
                self.issues.append(
                    f"data/{key} has leading length "
                    f"{array.shape[0] if array.ndim else 'scalar'}, expected {self.total_steps}"
                )
                continue
            self.aligned_keys.append(key)

        if "camera_rgb" not in self.aligned_keys:
            raise DatasetFormatError("expected aligned data/camera_rgb")
        camera = self.data["camera_rgb"]
        if camera.ndim != 4 or camera.shape[-1] != 3:
            raise DatasetFormatError(
                f"data/camera_rgb must have shape [T,H,W,3], found {camera.shape}"
            )

        self._graphable_keys = self._find_graphable_keys()
        self._source_episode_by_output = self._read_source_episode_mapping()

    @property
    def episode_count(self) -> int:
        return int(self.episode_ends.size)

    @property
    def aligned_hz(self) -> float:
        try:
            value = float(self.attrs.get("aligned_hz", 30.0))
        except (TypeError, ValueError):
            value = 30.0
        return value if np.isfinite(value) and value > 0 else 30.0

    def _validate_episode(self, episode: int) -> int:
        episode = int(episode)
        if episode < 0 or episode >= self.episode_count:
            raise IndexError(
                f"episode {episode} is outside [0, {max(0, self.episode_count - 1)}]"
            )
        return episode

    def episode_bounds(self, episode: int) -> tuple[int, int]:
        episode = self._validate_episode(episode)
        stop = int(self.episode_ends[episode])
        start = 0 if episode == 0 else int(self.episode_ends[episode - 1])
        return start, stop

    def episode_length(self, episode: int) -> int:
        start, stop = self.episode_bounds(episode)
        return stop - start

    def _episode_slice(
        self,
        episode: int,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
    ) -> tuple[slice, int, int, int]:
        episode_start, episode_stop = self.episode_bounds(episode)
        length = episode_stop - episode_start
        local_start = max(0, min(int(start), length))
        local_end = length if end is None else max(local_start, min(int(end), length))
        stride = max(1, int(stride))
        return (
            slice(episode_start + local_start, episode_start + local_end, stride),
            local_start,
            local_end,
            stride,
        )

    def timestamps(
        self,
        episode: int,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
    ) -> np.ndarray:
        slc, local_start, local_end, stride = self._episode_slice(
            episode, start, end, stride
        )
        episode_start, episode_stop = self.episode_bounds(episode)
        length = episode_stop - episode_start
        if length == 0:
            return np.asarray([], dtype=np.float64)

        if "timestamp" in self.aligned_keys:
            values = np.asarray(self.data["timestamp"][slc], dtype=np.float64)
            origin = float(self.data["timestamp"][episode_start])
        elif "timestamp_ns" in self.aligned_keys:
            values = np.asarray(self.data["timestamp_ns"][slc], dtype=np.float64) * 1e-9
            origin = float(self.data["timestamp_ns"][episode_start]) * 1e-9
        else:
            indices = np.arange(local_start, local_end, stride, dtype=np.float64)
            return indices / self.aligned_hz

        relative = values - origin
        if relative.size and not np.all(np.isfinite(relative)):
            indices = np.arange(local_start, local_end, stride, dtype=np.float64)
            return indices / self.aligned_hz
        return relative

    def _find_graphable_keys(self) -> list[str]:
        graphable: list[str] = []
        for key in self.aligned_keys:
            array = self.data[key]
            if key == "camera_rgb" or key in TIMESTAMP_KEYS:
                continue
            if array.ndim not in (1, 2):
                continue
            if array.dtype.kind not in {"b", "i", "u", "f"}:
                continue
            graphable.append(key)
        return graphable

    def _read_source_episode_mapping(self) -> dict[int, int]:
        clean = self.attrs.get("clean")
        if not isinstance(clean, dict):
            return {}
        segments = clean.get("segments")
        if not isinstance(segments, list) or len(segments) != self.episode_count:
            return {}
        mapping: dict[int, int] = {}
        for output_episode, segment in enumerate(segments):
            if not isinstance(segment, dict) or "source_episode" not in segment:
                continue
            try:
                mapping[output_episode] = int(segment["source_episode"])
            except (TypeError, ValueError):
                continue
        return mapping

    def _metadata_value(self, key: str, episode: int) -> int | None:
        if key not in self.meta:
            return None
        array = self.meta[key]
        if array.ndim != 1 or episode >= array.shape[0]:
            return None
        return int(array[episode])

    def _duration_and_fps(self, episode: int) -> tuple[float, float]:
        length = self.episode_length(episode)
        if length < 2:
            return 0.0, self.aligned_hz
        timestamps = self.timestamps(episode)
        if timestamps.size != length:
            return (length - 1) / self.aligned_hz, self.aligned_hz
        diffs = np.diff(timestamps)
        valid = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
        if valid.size < max(1, diffs.size // 2):
            return (length - 1) / self.aligned_hz, self.aligned_hz
        median_dt = float(np.median(valid))
        return float(timestamps[-1]), 1.0 / median_dt

    def _binary_count(self, key: str, episode: int) -> int | None:
        if key not in self.aligned_keys:
            return None
        start, stop = self.episode_bounds(episode)
        values = np.asarray(self.data[key][start:stop]).reshape(-1)
        return int(np.count_nonzero(values > 0))

    def episode_metrics(self, episode: int) -> dict[str, Any]:
        episode = self._validate_episode(episode)
        start, stop = self.episode_bounds(episode)
        length = stop - start
        duration, inferred_fps = self._duration_and_fps(episode)

        valid = self._binary_count("alignment_valid", episode)
        tracking = self._binary_count("tracking", episode)
        clutch = self._binary_count("clutch", episode)
        accepted = self._binary_count("motor_command_accepted", episode)
        grasp = self._binary_count("grasp_label", episode)

        active_steps: int | None = None
        if "action" in self.aligned_keys:
            action = np.asarray(self.data["action"][start:stop], dtype=np.float64)
            motion = (
                action[:, : min(6, action.shape[1])] if action.ndim == 2 else action
            )
            if motion.ndim == 1:
                active_steps = int(np.count_nonzero(np.abs(motion) > 1e-5))
            else:
                active_steps = int(
                    np.count_nonzero(np.linalg.norm(motion, axis=1) > 1e-5)
                )

        sequence_gaps: int | None = None
        if "camera_sequence_num" in self.aligned_keys and length >= 2:
            seq = np.asarray(
                self.data["camera_sequence_num"][start:stop], dtype=np.int64
            )
            sequence_gaps = int(np.count_nonzero(np.diff(seq) != 1))

        def ratio(count: int | None) -> float | None:
            if count is None or length == 0:
                return None
            return float(count / length)

        metadata_valid = self._metadata_value("episode_valid_steps", episode)
        metadata_invalid = self._metadata_value("episode_invalid_steps", episode)
        return {
            "episode_index": episode,
            "source_episode": self._source_episode_by_output.get(episode),
            "start": start,
            "end": stop,
            "length": length,
            "duration_sec": duration,
            "inferred_fps": inferred_fps,
            "valid_steps": valid,
            "valid_ratio": ratio(valid),
            "tracking_ratio": ratio(tracking),
            "clutch_ratio": ratio(clutch),
            "motor_accepted_ratio": ratio(accepted),
            "grasp_ratio": ratio(grasp),
            "active_motion_ratio": ratio(active_steps),
            "camera_sequence_gaps": sequence_gaps,
            "metadata_valid_steps": metadata_valid,
            "metadata_invalid_steps": metadata_invalid,
            "metadata_counts_match": (
                None
                if valid is None or metadata_valid is None or metadata_invalid is None
                else metadata_valid == valid and metadata_invalid == length - valid
            ),
        }

    def episodes(self) -> list[dict[str, Any]]:
        return [self.episode_metrics(episode) for episode in range(self.episode_count)]

    def summary(self, requested_episode: int | None = None) -> dict[str, Any]:
        camera = self.data["camera_rgb"]
        clean = self.attrs.get("clean")
        return {
            "input_path": str(self.path),
            "format": "giraf_zarr",
            "supported": True,
            "schema_version": self.attrs.get("schema_version"),
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "camera_shape": [int(value) for value in camera.shape[1:]],
            "aligned_hz": self.aligned_hz,
            "requested_episode": requested_episode,
            "cleaned": isinstance(clean, dict),
            "issues": list(self.issues),
        }

    def _channel_names(self, key: str, width: int) -> list[str]:
        configured: list[str] = []
        if key == "action":
            configured = [str(value) for value in self.attrs.get("action_fields", [])]
        elif key == "state":
            configured = [str(value) for value in self.attrs.get("state_fields", [])]
        elif key in {"joint_velocity_command", "joint_position_command"}:
            configured = [str(value) for value in self.attrs.get("joint_fields", [])]
        elif key == "can_position_target":
            configured = [
                str(value) for value in self.attrs.get("joint_fields", [])[:3]
            ]
        elif key == "dynamixel_target_ticks":
            joint_fields = [str(value) for value in self.attrs.get("joint_fields", [])]
            configured = joint_fields[3:] + ["grasp_target"]

        if len(configured) == width:
            return configured
        if width == 1:
            return [key]
        return [f"{key}[{index}]" for index in range(width)]

    @staticmethod
    def _group_for_key(key: str) -> str:
        if key == "state":
            return "State"
        if key in {
            "action",
            "joint_velocity_command",
            "joint_position_command",
            "can_position_target",
            "dynamixel_target_ticks",
        }:
            return "Commands"
        if key in {
            "alignment_valid",
            "tracking",
            "clutch",
            "motor_command_accepted",
            "grasp_label",
        }:
            return "Health"
        if key in MILLISECOND_KEYS or key == "camera_sequence_num":
            return "Timing"
        return "Core"

    def schema(self, episode: int) -> dict[str, Any]:
        episode = self._validate_episode(episode)
        start, stop = self.episode_bounds(episode)
        length = stop - start
        keys: list[dict[str, Any]] = []
        groups = {group: [] for group in GROUP_ORDER}

        for key in self._graphable_keys:
            array = self.data[key]
            width = 1 if array.ndim == 1 else int(array.shape[1])
            group = self._group_for_key(key)
            channel_names = self._channel_names(key, width)
            if key in MILLISECOND_KEYS:
                channel_names = [
                    name.removesuffix("_ns") + "_ms" for name in channel_names
                ]
            keys.append(
                {
                    "key": key,
                    "shape": [length, *[int(value) for value in array.shape[1:]]],
                    "dtype": str(array.dtype),
                    "channels": channel_names,
                    "group": group,
                }
            )
            groups[group].append(key)
            groups["All"].append(key)

        core = [
            key
            for key in ("action", "grasp_label", "alignment_valid")
            if key in self._graphable_keys
        ]
        groups["Core"] = core
        return {
            "episode_index": episode,
            "length": length,
            "start": start,
            "end": stop,
            "has_timestamps": "timestamp" in self.aligned_keys
            or "timestamp_ns" in self.aligned_keys,
            "cameras": [
                {
                    "stream_id": "main",
                    "rgb_key": "camera_rgb",
                    "rgb_shape": [
                        int(value) for value in self.data["camera_rgb"].shape[1:]
                    ],
                }
            ],
            "keys": keys,
            "key_groups": groups,
            "issues": list(self.issues),
        }

    def frame(self, episode: int, index: int) -> np.ndarray:
        episode_start, episode_stop = self.episode_bounds(episode)
        length = episode_stop - episode_start
        index = int(index)
        if index < 0 or index >= length:
            raise IndexError(f"frame {index} is outside [0, {max(0, length - 1)}]")
        return np.asarray(self.data["camera_rgb"][episode_start + index])

    def signal_payload(
        self,
        episode: int,
        keys: list[str],
        start: int,
        end: int,
        stride: int = 1,
    ) -> dict[str, Any]:
        slc, local_start, local_end, stride = self._episode_slice(
            episode, start, end, stride
        )
        indices = np.arange(local_start, local_end, stride, dtype=np.int64)
        timestamps = self.timestamps(episode, local_start, local_end, stride)
        allowed = set(self._graphable_keys)
        series: dict[str, list[dict[str, Any]]] = {}
        skipped: list[str] = []

        for key in keys:
            if key not in allowed:
                skipped.append(key)
                continue
            values = np.asarray(self.data[key][slc])
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2:
                skipped.append(key)
                continue
            values = values.astype(np.float64)
            if key in MILLISECOND_KEYS:
                values *= 1e-6
            names = self._channel_names(key, values.shape[1])
            if key in MILLISECOND_KEYS:
                names = [name.removesuffix("_ns") + "_ms" for name in names]
            series[key] = [
                {"name": names[channel], "values": values[:, channel].tolist()}
                for channel in range(values.shape[1])
            ]

        return {
            "episode_index": int(episode),
            "start": local_start,
            "end": local_end,
            "stride": stride,
            "indices": indices.tolist(),
            "timestamps": timestamps.tolist(),
            "series": series,
            "skipped": skipped,
        }

    def events(self, episode: int, max_events: int = 500) -> list[dict[str, Any]]:
        episode = self._validate_episode(episode)
        start, stop = self.episode_bounds(episode)
        length = stop - start
        if length == 0:
            return []
        timestamps = self.timestamps(episode)
        events: list[dict[str, Any]] = []

        def add(
            index: int, event_type: str, label: str, details: dict[str, Any]
        ) -> None:
            if len(events) >= max_events:
                return
            events.append(
                {
                    "idx": int(index),
                    "time": float(timestamps[index])
                    if index < timestamps.size
                    else 0.0,
                    "type": event_type,
                    "label": label,
                    "details": details,
                }
            )

        transition_specs = (
            ("alignment_valid", "validity", "Valid", "INVALID"),
            ("clutch", "clutch", "Clutch ON", "Clutch OFF"),
            ("tracking", "tracking", "Tracking ON", "Tracking OFF"),
            ("grasp_label", "grasp", "Grasp ON", "Grasp OFF"),
            (
                "motor_command_accepted",
                "motor",
                "Motor command accepted",
                "Motor command rejected",
            ),
        )
        for key, event_type, on_label, off_label in transition_specs:
            if key not in self.aligned_keys:
                continue
            values = np.asarray(self.data[key][start:stop]).reshape(-1) > 0
            if values.size == 0:
                continue
            if key == "alignment_valid" and not bool(values[0]):
                add(0, event_type, off_label, {"key": key, "value": 0})
            transitions = np.flatnonzero(values[1:] != values[:-1]) + 1
            for index in transitions.tolist():
                enabled = bool(values[index])
                add(
                    index,
                    event_type,
                    on_label if enabled else off_label,
                    {"key": key, "value": int(enabled)},
                )

        if "camera_sequence_num" in self.aligned_keys and length >= 2:
            sequence = np.asarray(
                self.data["camera_sequence_num"][start:stop], dtype=np.int64
            )
            for index in (np.flatnonzero(np.diff(sequence) != 1) + 1).tolist():
                add(
                    index,
                    "camera_gap",
                    "Camera sequence gap",
                    {
                        "previous": int(sequence[index - 1]),
                        "current": int(sequence[index]),
                    },
                )

        if "action" in self.aligned_keys and length >= 3:
            action = np.asarray(self.data["action"][start:stop], dtype=np.float64)
            if action.ndim == 2:
                differences = np.linalg.norm(np.diff(action, axis=0), axis=1)
                standard_deviation = float(np.std(differences))
                if standard_deviation > 1e-12:
                    threshold = float(np.mean(differences) + 4.0 * standard_deviation)
                    for index in (np.flatnonzero(differences > threshold) + 1).tolist():
                        add(
                            index,
                            "action_jump",
                            "Action jump",
                            {
                                "magnitude": float(differences[index - 1]),
                                "threshold": threshold,
                            },
                        )

        return sorted(events, key=lambda item: (item["idx"], item["type"]))[:max_events]
