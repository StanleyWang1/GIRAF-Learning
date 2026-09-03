from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import zarr

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.key_registry import build_key_groups, infer_group
from dataset_visualization.backend.types import CameraStream, DatasetSummary, EpisodeSchema, KeyInfo


class _VideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._last_idx = -1
        self._lock = Lock()
        self.sequential_hits = 0
        self.seek_reads = 0

    def read(self, idx: int) -> np.ndarray:
        if self.frame_count <= 0:
            raise RuntimeError(f"No frames in video: {self.video_path}")

        target = max(0, min(self.frame_count - 1, int(idx)))

        with self._lock:
            if target != self._last_idx + 1:
                self.seek_reads += 1
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            else:
                self.sequential_hits += 1
            ok, frame_bgr = self._cap.read()
            if not ok:
                # Retry once with explicit seek.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame_bgr = self._cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {target} from {self.video_path}")

            self._last_idx = target
            return frame_bgr

    def close(self) -> None:
        with self._lock:
            self._cap.release()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "sequential_hits": int(self.sequential_hits),
                "seek_reads": int(self.seek_reads),
            }


class RawSidecarAdapter(BaseAdapter):
    format_name = "raw_sidecar"

    def __init__(self, input_path: Path, videos_root: Optional[Path] = None, depth_shape: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.input_path = Path(input_path)
        self.root = zarr.open(str(self.input_path), mode="r")
        self.data_group = self.root["data"] if "data" in self.root else self.root
        self.meta_group = self.root.get("meta")
        self.manual_depth_shape = depth_shape

        default_root = self.input_path.parent / f"{self.input_path.stem}_videos"
        self.videos_root = Path(videos_root) if videos_root is not None else default_root

        self._episode_ends = self._load_episode_ends()
        self._keys = sorted(list(self.data_group.keys()))
        self._stream_cache: Dict[int, Dict[str, CameraStream]] = {}

        self._video_readers: Dict[Tuple[int, str], _VideoReader] = {}
        self._reader_lock = Lock()

    def _load_episode_ends(self) -> np.ndarray:
        if self.meta_group is not None and "episode_ends" in self.meta_group:
            return np.asarray(self.meta_group["episode_ends"][:], dtype=np.int64)

        if len(self.data_group.keys()) == 0:
            return np.asarray([], dtype=np.int64)

        first_key = next(iter(self.data_group.keys()))
        n_steps = int(self.data_group[first_key].shape[0])
        return np.asarray([n_steps], dtype=np.int64)

    def _validate_episode_index(self, episode_index: int) -> None:
        if episode_index < 0 or episode_index >= self.episode_count():
            raise IndexError(f"episode index {episode_index} out of range")

    def _episode_dir(self, episode_index: int) -> Path:
        return self.videos_root / f"ep_{episode_index}"

    def _scan_episode_streams(self, episode_index: int) -> Dict[str, CameraStream]:
        if episode_index in self._stream_cache:
            return self._stream_cache[episode_index]

        streams: Dict[str, CameraStream] = {}
        ep_dir = self._episode_dir(episode_index)
        if not ep_dir.exists():
            self._stream_cache[episode_index] = streams
            return streams

        rgb_files = sorted(ep_dir.glob("camera_*_rgb.mp4"))
        depth_files = sorted(ep_dir.glob("camera_*_depth.bin"))

        for rgb_path in rgb_files:
            serial = rgb_path.stem.replace("camera_", "").replace("_rgb", "")
            stream = streams.get(serial)
            if stream is None:
                stream = CameraStream(stream_id=serial)
                streams[serial] = stream
            stream.rgb_source = "sidecar"
            stream.rgb_path = rgb_path
            stream.rgb_shape = self._video_shape(rgb_path)

        start, end = self.episode_bounds(episode_index)
        ep_len = end - start

        for depth_path in depth_files:
            serial = depth_path.stem.replace("camera_", "").replace("_depth", "")
            stream = streams.get(serial)
            if stream is None:
                stream = CameraStream(stream_id=serial)
                streams[serial] = stream
            stream.depth_source = "sidecar"
            stream.depth_path = depth_path
            stream.depth_shape = self._infer_depth_shape(depth_path, ep_len, stream.rgb_shape)

        streams = dict(sorted(streams.items(), key=lambda kv: kv[0]))
        self._stream_cache[episode_index] = streams
        return streams

    @staticmethod
    def _video_shape(video_path: Path) -> Optional[Tuple[int, int, int]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if h <= 0 or w <= 0:
            return None
        return (h, w, 3)

    def _infer_depth_shape(
        self,
        depth_path: Path,
        expected_length: int,
        rgb_shape: Optional[Tuple[int, int, int]],
    ) -> Optional[Tuple[int, int]]:
        if self.manual_depth_shape is not None:
            return self.manual_depth_shape

        file_size = depth_path.stat().st_size
        if file_size <= 0:
            return None

        candidates: List[Tuple[int, int]] = []
        if rgb_shape is not None:
            h, w, _ = rgb_shape
            candidates.extend(
                [
                    (h, w),
                    (max(1, h // 2), max(1, w // 2)),
                    (max(1, h // 4), max(1, w // 4)),
                ]
            )

        candidates.extend([(90, 160), (180, 320), (224, 224), (360, 640), (480, 640)])

        seen = set()
        ordered_candidates: List[Tuple[int, int]] = []
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            ordered_candidates.append(c)

        best_shape: Optional[Tuple[int, int]] = None
        best_score = float("inf")

        for h, w in ordered_candidates:
            frame_bytes = 2 * h * w
            if frame_bytes <= 0:
                continue
            if file_size % frame_bytes != 0:
                continue
            frame_count = file_size // frame_bytes
            score = abs(int(frame_count) - int(expected_length))
            tie_break = h * w
            candidate_score = score * 10_000_000 + tie_break
            if candidate_score < best_score:
                best_score = candidate_score
                best_shape = (h, w)

        return best_shape

    def _get_reader(self, episode_index: int, stream_id: str) -> _VideoReader:
        key = (episode_index, stream_id)
        with self._reader_lock:
            reader = self._video_readers.get(key)
            if reader is not None:
                return reader

            streams = self._scan_episode_streams(episode_index)
            stream = streams.get(stream_id)
            if stream is None or stream.rgb_path is None:
                raise KeyError(f"RGB stream not found: {stream_id}")

            reader = _VideoReader(stream.rgb_path)
            self._video_readers[key] = reader
            return reader

    def close(self) -> None:
        with self._reader_lock:
            readers = list(self._video_readers.values())
            self._video_readers.clear()
        for reader in readers:
            reader.close()

    def reader_stats(self) -> Dict[str, Dict[str, int]]:
        with self._reader_lock:
            out: Dict[str, Dict[str, int]] = {}
            for (episode_index, stream_id), reader in self._video_readers.items():
                out[f"{episode_index}:{stream_id}"] = reader.stats()
            return out

    def dataset_summary(self) -> DatasetSummary:
        total_steps = int(self._episode_ends[-1]) if len(self._episode_ends) > 0 else 0
        modalities: List[str] = []
        issues: List[str] = []

        if self.videos_root.exists():
            modalities.append("rgb")
        else:
            issues.append(f"Videos root not found: {self.videos_root}")

        if any("tau" in k for k in self._keys):
            modalities.append("torque")
        if any("wrench" in k for k in self._keys):
            modalities.append("wrench")
        if any(k.startswith("robot0_eef_pos") or k.startswith("robot1_eef_pos") for k in self._keys):
            modalities.append("eef")

        if self.episode_count() > 0:
            streams = self._scan_episode_streams(0)
            if any(s.depth_path is not None for s in streams.values()):
                modalities.append("depth")

        return DatasetSummary(
            input_path=str(self.input_path),
            format=self.format_name,
            supported=True,
            unsupported_reason=None,
            episode_count=self.episode_count(),
            total_steps=total_steps,
            available_modalities=sorted(set(modalities)),
            issues=issues,
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
        _ = episode_index
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
        streams = self._scan_episode_streams(episode_index)
        graphable = set(self.graphable_keys(episode_index))

        total_steps = int(self._episode_ends[-1]) if len(self._episode_ends) else 0
        key_infos: List[KeyInfo] = []
        for key in self._keys:
            arr = self.data_group[key]
            shape = list(arr.shape)
            if shape and shape[0] == total_steps:
                shape[0] = ep_len
            key_infos.append(
                KeyInfo(
                    key=key,
                    shape=shape,
                    dtype=str(arr.dtype),
                    graphable=key in graphable,
                    group=infer_group(key),
                )
            )

        issues: List[str] = []
        if len(streams) == 0:
            issues.append(f"No camera streams found under {self._episode_dir(episode_index)}")

        key_groups = build_key_groups(self._keys, graphable)

        return EpisodeSchema(
            episode_index=episode_index,
            length=ep_len,
            start=start,
            end=end,
            has_timestamps="timestamps" in self._keys,
            cameras=list(streams.values()),
            keys=key_infos,
            key_groups=key_groups,
            issues=issues,
        )

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
        streams = self._scan_episode_streams(episode_index)
        return list(streams.keys())

    def has_depth(self, episode_index: int, stream_id: str) -> bool:
        streams = self._scan_episode_streams(episode_index)
        stream = streams.get(stream_id)
        return stream is not None and stream.depth_path is not None

    def frame_rgb(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        reader = self._get_reader(episode_index, stream_id)
        frame_bgr = reader.read(idx)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def frame_depth(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        streams = self._scan_episode_streams(episode_index)
        stream = streams.get(stream_id)
        if stream is None or stream.depth_path is None:
            raise KeyError(f"Depth stream not found: {stream_id}")

        depth_shape = stream.depth_shape
        if depth_shape is None:
            raise RuntimeError(f"Unable to infer depth shape for {stream.depth_path}")

        h, w = depth_shape
        frame_bytes = 2 * h * w
        file_size = stream.depth_path.stat().st_size
        if file_size < frame_bytes:
            raise RuntimeError(f"Depth file too small: {stream.depth_path}")

        frame_count = max(1, file_size // frame_bytes)
        idx = max(0, min(frame_count - 1, int(idx)))

        with open(stream.depth_path, "rb") as f:
            f.seek(idx * frame_bytes)
            raw = f.read(frame_bytes)
            if len(raw) != frame_bytes:
                f.seek((frame_count - 1) * frame_bytes)
                raw = f.read(frame_bytes)
                if len(raw) != frame_bytes:
                    raise RuntimeError(f"Failed reading depth frame from {stream.depth_path}")

        frame = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
        return frame
