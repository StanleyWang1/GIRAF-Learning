from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import zarr

from dataset_visualization.backend.adapters.base import BaseAdapter
from dataset_visualization.backend.adapters.full_zarr import FullZarrAdapter
from dataset_visualization.backend.adapters.raw_sidecar import RawSidecarAdapter
from dataset_visualization.backend.key_registry import parse_camera_key
from dataset_visualization.backend.types import DatasetSummary, EpisodeSchema


def _read_data_keys(input_path: Path) -> List[str]:
    root = zarr.open(str(input_path), mode="r")
    data_group = root["data"] if "data" in root else root
    return sorted(list(data_group.keys()))


class UnsupportedAdapter(BaseAdapter):
    format_name = "unsupported"

    def __init__(self, input_path: Path, reason: str):
        super().__init__()
        self.input_path = str(input_path)
        self.unsupported_reason = reason

    def dataset_summary(self) -> DatasetSummary:
        return DatasetSummary(
            input_path=self.input_path,
            format=self.format_name,
            supported=False,
            unsupported_reason=self.unsupported_reason,
            episode_count=0,
            total_steps=0,
            available_modalities=[],
            issues=[self.unsupported_reason or "Unsupported dataset"],
        )

    def episode_count(self) -> int:
        return 0

    def episode_bounds(self, episode_index: int) -> Tuple[int, int]:
        raise RuntimeError(self.unsupported_reason)

    def episode_schema(self, episode_index: int) -> EpisodeSchema:
        raise RuntimeError(self.unsupported_reason)

    def episode_timestamps(self, episode_index: int, start: int, end: int, stride: int = 1) -> np.ndarray:
        raise RuntimeError(self.unsupported_reason)

    def signal_window(self, episode_index: int, key: str, start: int, end: int, stride: int = 1) -> np.ndarray:
        raise RuntimeError(self.unsupported_reason)

    def frame_rgb(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        raise RuntimeError(self.unsupported_reason)

    def frame_depth(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        raise RuntimeError(self.unsupported_reason)

    def list_stream_ids(self, episode_index: int) -> List[str]:
        return []

    def has_depth(self, episode_index: int, stream_id: str) -> bool:
        return False

    def graphable_keys(self, episode_index: int) -> List[str]:
        return []

    def all_keys(self) -> List[str]:
        return []


def create_adapter(
    input_path: Path,
    videos_root: Optional[Path] = None,
    depth_shape: Optional[Tuple[int, int]] = None,
) -> BaseAdapter:
    keys = _read_data_keys(input_path)

    index_keys = [k for k in keys if re.match(r"^camera_.+_indices$", k)]
    if index_keys:
        return UnsupportedAdapter(
            input_path,
            (
                "Index-based zarr format (camera_*_indices) is not supported in v1. "
                "Convert/merge to full zarr or use raw zarr + sidecar videos."
            ),
        )

    has_inline_camera = any(parse_camera_key(k) is not None for k in keys)
    if has_inline_camera:
        return FullZarrAdapter(input_path)

    return RawSidecarAdapter(input_path, videos_root=videos_root, depth_shape=depth_shape)
