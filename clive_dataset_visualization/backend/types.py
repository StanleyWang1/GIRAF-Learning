from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


MediaSource = Literal["zarr", "sidecar"]


@dataclass
class CameraStream:
    stream_id: str
    rgb_source: Optional[MediaSource] = None
    depth_source: Optional[MediaSource] = None
    rgb_key: Optional[str] = None
    depth_key: Optional[str] = None
    rgb_path: Optional[Path] = None
    depth_path: Optional[Path] = None
    rgb_shape: Optional[Tuple[int, int, int]] = None
    depth_shape: Optional[Tuple[int, int]] = None


@dataclass
class KeyInfo:
    key: str
    shape: List[int]
    dtype: str
    graphable: bool
    group: str


@dataclass
class EpisodeSchema:
    episode_index: int
    length: int
    start: int
    end: int
    has_timestamps: bool
    cameras: List[CameraStream] = field(default_factory=list)
    keys: List[KeyInfo] = field(default_factory=list)
    key_groups: Dict[str, List[str]] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class DatasetSummary:
    input_path: str
    format: str
    supported: bool
    unsupported_reason: Optional[str]
    episode_count: int
    total_steps: int
    available_modalities: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
