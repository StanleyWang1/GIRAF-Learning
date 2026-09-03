from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataset_visualization.backend.types import DatasetSummary, EpisodeSchema


class BaseAdapter(ABC):
    format_name: str = "unknown"

    def __init__(self) -> None:
        self.unsupported_reason: Optional[str] = None

    @property
    def supported(self) -> bool:
        return self.unsupported_reason is None

    @abstractmethod
    def dataset_summary(self) -> DatasetSummary:
        raise NotImplementedError

    @abstractmethod
    def episode_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def episode_bounds(self, episode_index: int) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def episode_schema(self, episode_index: int) -> EpisodeSchema:
        raise NotImplementedError

    @abstractmethod
    def episode_timestamps(self, episode_index: int, start: int, end: int, stride: int = 1) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def signal_window(self, episode_index: int, key: str, start: int, end: int, stride: int = 1) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def frame_rgb(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def frame_depth(self, episode_index: int, stream_id: str, idx: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def list_stream_ids(self, episode_index: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def has_depth(self, episode_index: int, stream_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def graphable_keys(self, episode_index: int) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def all_keys(self) -> List[str]:
        raise NotImplementedError
