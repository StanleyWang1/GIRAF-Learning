from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.key_registry import build_key_groups, infer_group, parse_camera_key
from dataset_visualization.backend.types import CameraStream, DatasetSummary, EpisodeSchema, KeyInfo


class FullZarrAdapter(BaseAdapter):
    format_name = "full_zarr"

    def __init__(self, input_path: Path):
        super().__init__()
        self.input_path = Path(input_path)
        self.root = zarr.open(str(self.input_path), mode="r")
        self.data_group = self.root["data"] if "data" in self.root else self.root
        self.meta_group = self.root.get("meta")

        self._episode_ends = self._load_episode_ends()
        self._keys = sorted(list(self.data_group.keys()))
        self._stream_map = self._build_stream_map()

    def _load_episode_ends(self) -> np.ndarray:
        if self.meta_group is not None and "episode_ends" in self.meta_group:
            return np.asarray(self.meta_group["episode_ends"][:], dtype=np.int64)

        # Fallback: single episode from first key.
        if len(self.data_group.keys()) == 0:
            return np.asarray([], dtype=np.int64)

        first_key = next(iter(self.data_group.keys()))
        n_steps = int(self.data_group[first_key].shape[0])
        return np.asarray([n_steps], dtype=np.int64)

    def _build_stream_map(self) -> Dict[str, CameraStream]:
        streams: Dict[str, CameraStream] = {}
        for key in self._keys:
            parsed = parse_camera_key(key)
            if parsed is None:
                continue
            stream_id, modality = parsed
            stream = streams.get(stream_id)
            if stream is None:
                stream = CameraStream(stream_id=stream_id)
                streams[stream_id] = stream

            arr = self.data_group[key]
            if modality == "rgb":
                stream.rgb_source = "zarr"
                stream.rgb_key = key
                if len(arr.shape) >= 4:
                    stream.rgb_shape = (int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
            elif modality == "depth":
                stream.depth_source = "zarr"
                stream.depth_key = key
                if len(arr.shape) >= 3:
                    stream.depth_shape = (int(arr.shape[1]), int(arr.shape[2]))

        return dict(sorted(streams.items(), key=lambda kv: kv[0]))

    def _validate_episode_index(self, episode_index: int) -> None:
        if episode_index < 0 or episode_index >= self.episode_count():
            raise IndexError(f"episode index {episode_index} out of range")

    def _global_slice(self, episode_index: int, start: int, end: int, stride: int = 1) -> slice:
        ep_start, ep_end = self.episode_bounds(episode_index)
        ep_len = ep_end - ep_start

        if start < 0:
            start = 0
        if end < 0:
            end = 0
        if start > ep_len:
            start = ep_len
        if end > ep_len:
            end = ep_len
        if end < start:
            end = start
        if stride <= 0:
            stride = 1

        return slice(ep_start + start, ep_start + end, stride)

    def dataset_summary(self) -> DatasetSummary:
        total_steps = int(self._episode_ends[-1]) if len(self._episode_ends) > 0 else 0
        modalities: List[str] = []
        if any(k.startswith("camera_") and k.endswith("_rgb") for k in self._keys):
            modalities.append("rgb")
        if any(k.startswith("camera_") and k.endswith("_depth") for k in self._keys):
            modalities.append("depth")
        if any("wrench" in k for k in self._keys):
            modalities.append("wrench")
        if any("tau" in k for k in self._keys):
            modalities.append("torque")
        if any(k.startswith("robot0_eef_pos") or k.startswith("robot1_eef_pos") for k in self._keys):
            modalities.append("eef")

        return DatasetSummary(
            input_path=str(self.input_path),
            format=self.format_name,
            supported=True,
            unsupported_reason=None,
            episode_count=self.episode_count(),
            total_steps=total_steps,
            available_modalities=sorted(set(modalities)),
            issues=[],
        )

    def episode_count(self) -> int:
        return int(len(self._episode_ends))

    def episode_bounds(self, episode_index: int) -> Tuple[int, int]:
        self._validate_episode_index(episode_index)
        end = int(self._episode_ends[episode_index])
        start = 0 if episode_index == 0 else int(self._episode_ends[episode_index - 1])
        return start, end

    def all_keys(self) -> List[str]:
        return list(self._keys)

    def graphable_keys(self, episode_index: int) -> List[str]:
        _ = episode_index  # same key set across episodes.
        graphable: List[str] = []
        total_steps = int(self._episode_ends[-1]) if len(self._episode_ends) else 0
        for key in self._keys:
            arr = self.data_group[key]
            if len(arr.shape) == 0:
                continue
            if arr.shape[0] != total_steps:
                continue
            if len(arr.shape) > 2:
                continue
            if arr.dtype.kind not in {"i", "u", "f", "b"}:
                continue
            graphable.append(key)
        return graphable

    def episode_schema(self, episode_index: int) -> EpisodeSchema:
        start, end = self.episode_bounds(episode_index)
        ep_len = end - start
        graphable = set(self.graphable_keys(episode_index))

        keys_info: List[KeyInfo] = []
        total_steps = int(self._episode_ends[-1]) if len(self._episode_ends) else 0
        for key in self._keys:
            arr = self.data_group[key]
            shape = list(arr.shape)
            if shape and shape[0] == total_steps:
                shape[0] = ep_len
            keys_info.append(
                KeyInfo(
                    key=key,
                    shape=shape,
                    dtype=str(arr.dtype),
                    graphable=key in graphable,
                    group=infer_group(key),
                )
            )

        key_groups = build_key_groups(self._keys, graphable)

        return EpisodeSchema(
            episode_index=episode_index,
            length=ep_len,
            start=start,
            end=end,
            has_timestamps="timestamps" in self._keys,
            cameras=list(self._stream_map.values()),
            keys=keys_info,
            key_groups=key_groups,
            issues=[],
        )

    def episode_timestamps(self, episode_index: int, start: int, end: int, stride: int = 1) -> np.ndarray:
        if "timestamps" not in self.data_group:
            if end <= start:
                return np.asarray([], dtype=np.float64)
            return np.arange(start, end, stride, dtype=np.float64)

        slc = self._global_slice(episode_index, start, end, stride)
        return np.asarray(self.data_group["timestamps"][slc])

    def signal_window(self, episode_index: int, key: str, start: int, end: int, stride: int = 1) -> np.ndarray:
        if key not in self.data_group:
            raise KeyError(f"unknown key: {key}")
        slc = self._global_slice(episode_index, start, end, stride)
        return np.asarray(self.data_group[key][slc])

    def list_stream_ids(self, episode_index: int) -> List[str]:
        _ = episode_index
        return list(self._stream_map.keys())

    def has_depth(self, episode_index: int, stream_id: str) -> bool:
        _ = episode_index
        stream = self._stream_map.get(stream_id)
        return stream is not None and stream.depth_key is not None

    def frame_rgb(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        stream = self._stream_map.get(stream_id)
        if stream is None or stream.rgb_key is None:
            raise KeyError(f"RGB stream not found: {stream_id}")

        start, end = self.episode_bounds(episode_index)
        ep_len = end - start
        if ep_len <= 0:
            raise ValueError("episode has no frames")
        idx = max(0, min(ep_len - 1, int(idx)))
        global_idx = start + idx

        frame = np.asarray(self.data_group[stream.rgb_key][global_idx])
        if frame.ndim != 3:
            raise ValueError(f"expected RGB frame with 3 dims, got shape={frame.shape}")
        return frame

    def frame_depth(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        stream = self._stream_map.get(stream_id)
        if stream is None or stream.depth_key is None:
            raise KeyError(f"Depth stream not found: {stream_id}")

        start, end = self.episode_bounds(episode_index)
        ep_len = end - start
        if ep_len <= 0:
            raise ValueError("episode has no frames")
        idx = max(0, min(ep_len - 1, int(idx)))
        global_idx = start + idx

        frame = np.asarray(self.data_group[stream.depth_key][global_idx])
        if frame.ndim != 2:
            raise ValueError(f"expected depth frame with 2 dims, got shape={frame.shape}")
        return frame
