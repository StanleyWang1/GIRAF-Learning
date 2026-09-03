from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_HISTORY_FRAMES = 180
DEFAULT_PREFETCH = 60
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


@dataclass(frozen=True)
class AppConfig:
    input_path: Path
    videos_root: Optional[Path]
    episode: Optional[int]
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    history_frames: int = DEFAULT_HISTORY_FRAMES
    prefetch: int = DEFAULT_PREFETCH
    depth_shape: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class CliArgs:
    input_path: Path
    videos_root: Optional[Path]
    episode: Optional[int]
    host: str
    port: int
    history_frames: int
    prefetch: int
    depth_shape: Optional[Tuple[int, int]]


def parse_depth_shape(values: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if values is None:
        return None
    h, w = values
    if h <= 0 or w <= 0:
        raise ValueError("depth-shape values must be > 0")
    return int(h), int(w)
